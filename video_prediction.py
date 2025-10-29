import os
import torch
import cv2
import numpy as np
from torchvision.utils import save_image
from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
from transformers.video_utils import load_video

# ---------------------
# 설정
# ---------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 직접 넣을 로컬 비디오 경로 (수정해서 쓰면 됨)
VIDEO_PATH = "/mnt/Inwoong/sam2/change.mp4"

OVERLAY_VIDEO_PATH = os.path.join(OUTPUT_DIR, "overlay_video.mp4")
ALL_MASKS_PATH = os.path.join(OUTPUT_DIR, "video_segments.pt")

# ---------------------
# 유틸 함수들
# ---------------------

def squeeze_mask_to_hw(mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    SAM2 post_process_masks 결과:
    보통 [B, N, H, W] or [N, H, W] or [H, W].
    이걸 [H, W] float32(0~1)로 압축.
    """
    mt = mask_tensor
    while mt.dim() > 2:
        mt = mt[0]
    return mt.float()  # [H,W]

def save_mask_tensor(mask_tensor: torch.Tensor, frame_idx: int, output_dir: str):
    """
    디버깅용으로 마스크를 PNG로 저장.
    mask_tensor는 아무 shape이든 가능.
    """
    hw_mask = squeeze_mask_to_hw(mask_tensor)  # [H,W]
    mask_to_save = hw_mask.unsqueeze(0).cpu()  # [1,H,W] for save_image
    out_path = os.path.join(output_dir, f"mask_{frame_idx:05d}.png")
    save_image(mask_to_save, out_path)
    print(f"[save mask] {out_path}")

def choose_dtype(device: str):
    """
    디바이스에 맞춰 안전한 dtype 고르기.
    - CUDA Ampere(major>=8) 이상이면 bfloat16
    - CUDA 구형이면 float16
    - CPU/mps 등은 float32
    """
    if device == "cpu":
        return torch.float32

    if device.startswith("cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16

    return torch.float32  # mps 등

def overlay_mask_on_frame_bgr(frame_bgr: np.ndarray,
                              mask_tensor: torch.Tensor,
                              alpha: float = 0.5) -> np.ndarray:
    """
    frame_bgr: (H,W,3) uint8 (BGR, OpenCV 형식)
    mask_tensor: SAM2 마스크 텐서 (아직 squeeze 전)
    alpha: 덮을 투명도 (0~1)
    return: overlay된 BGR uint8 프레임
    """
    # 1) [H,W] float(0~1)
    hw_mask = squeeze_mask_to_hw(mask_tensor)        # [H,W] float
    hw_mask_np = hw_mask.cpu().numpy()

    # 2) threshold 0.5 이상이면 해당 픽셀 살린다
    bin_mask = (hw_mask_np > 0.5).astype(np.uint8)   # [H,W] {0,1}

    # 3) 3채널 boolean mask로 확장
    bin_mask_3c = np.stack([bin_mask]*3, axis=-1).astype(bool)  # [H,W,3] bool

    # 4) 빨간색 (BGR에서 빨강은 (0,0,255))
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[..., 2] = 255

    # 5) 알파 블렌딩
    overlay = frame_bgr.copy()
    overlay[bin_mask_3c] = (
        frame_bgr[bin_mask_3c].astype(np.float32) * (1.0 - alpha)
        + color_layer[bin_mask_3c].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay

def main():
    # ---------------------
    # 1) 디바이스 / dtype
    # ---------------------
    device = infer_device()
    dtype = choose_dtype(device)

    # ---------------------
    # 2) 모델 / 프로세서 로드
    # ---------------------
    ckpt = "facebook/sam2.1-hiera-tiny"
    model = (
        Sam2VideoModel
        .from_pretrained(ckpt)
        .to(device, dtype=dtype)
    )
    processor = Sam2VideoProcessor.from_pretrained(ckpt)

    # ---------------------
    # 3) 비디오 로드 (로컬 파일)
    # ---------------------
    # OpenCV backend → frames는 BGR uint8, meta는 VideoMetadata (fps 등)
    video_frames_bgr, meta = load_video(VIDEO_PATH, backend="opencv")

    # fps 얻기
    if hasattr(meta, "fps"):
        fps = float(meta.fps)
    else:
        fps = 30.0

    # 사이즈
    h, w, _ = video_frames_bgr[0].shape
    print(f"[video] {len(video_frames_bgr)} frames, size {w}x{h}, fps={fps}")

    # SAM2는 RGB 이미지 기준으로 학습됐으므로 RGB로 변환해서 session에 넣자.
    video_frames_rgb = [
        cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB) for f_bgr in video_frames_bgr
    ]

    # ---------------------
    # 4) SAM2 비디오 세션 초기화 (RGB 프레임으로)
    # ---------------------
    inference_session = processor.init_video_session(
        video=video_frames_rgb,
        inference_device=device,
        dtype=dtype,
    )

    # ---------------------
    # 5) 첫 프레임에 클릭 프롬프트로 타겟 지정
    # ---------------------
    ann_frame_idx = 0
    ann_obj_id = 1  # 우리가 붙일 object 아이디

    # positive click: (x=210, y=350) 픽셀 위치
    # SAM2 API는 이런 nested 리스트 형태를 요구함
    points = [[[[200, 360]]]]
    labels = [[[1]]]  # 1 = positive click

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=points,
        input_labels=labels,
    )

    # ---------------------
    # 6) 첫 프레임 세그멘테이션
    # ---------------------
    first_out = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
    )

    first_mask = processor.post_process_masks(
        [first_out.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]  # e.g. [1,1,H,W]

    print(f"[frame {ann_frame_idx}] mask shape: {first_mask.shape}")
    save_mask_tensor(first_mask, ann_frame_idx, OUTPUT_DIR)

    # 결과 저장용 dict
    video_segments = {}
    overlay_frames = {}

    # 첫 프레임 overlay
    overlay_frames[ann_frame_idx] = overlay_mask_on_frame_bgr(
        frame_bgr=video_frames_bgr[ann_frame_idx],
        mask_tensor=first_mask,
        alpha=0.5,
    )
    video_segments[ann_frame_idx] = squeeze_mask_to_hw(first_mask).cpu()

    # ---------------------
    # 7) 나머지 프레임 추적 propagate
    # ---------------------
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        frame_idx_cur = sam2_video_output.frame_idx

        cur_mask = processor.post_process_masks(
            [sam2_video_output.pred_masks],
            original_sizes=[[inference_session.video_height, inference_session.video_width]],
            binarize=False,
        )[0]

        save_mask_tensor(cur_mask, frame_idx_cur, OUTPUT_DIR)

        overlay_frames[frame_idx_cur] = overlay_mask_on_frame_bgr(
            frame_bgr=video_frames_bgr[frame_idx_cur],
            mask_tensor=cur_mask,
            alpha=0.5,
        )
        video_segments[frame_idx_cur] = squeeze_mask_to_hw(cur_mask).cpu()

    # ---------------------
    # 8) overlay 영상 mp4로 저장 (BGR 그대로)
    # ---------------------
    sorted_idx = sorted(overlay_frames.keys())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OVERLAY_VIDEO_PATH,
        fourcc,
        fps if fps > 0 else 30.0,
        (w, h),
    )

    for idx in sorted_idx:
        writer.write(overlay_frames[idx])
    writer.release()
    print(f"[video overlay] saved overlay video to {OVERLAY_VIDEO_PATH}")

    # ---------------------
    # 9) 전체 마스크 텐서 dict 세이브
    # ---------------------
    torch.save(video_segments, ALL_MASKS_PATH)
    print(f"[masks dict] saved to {ALL_MASKS_PATH}")

    # ---------------------
    # 10) 요약
    # ---------------------
    print(f"[input video ] {VIDEO_PATH}")
    print(f"[overlay video] {OVERLAY_VIDEO_PATH}")
    print(f"[frames saved ] {len(sorted_idx)} frames")

if __name__ == "__main__":
    main()
