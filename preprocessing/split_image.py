import os
import shutil
import math

# ==========================================
# ⚙️ 설정 (여기를 본인 환경에 맞게 수정하세요)
# ==========================================
SOURCE_DIR = "../images"       # 압축 푼 원본 이미지들이 들어있는 폴더 경로
OUTPUT_BASE_DIR = "./data"    # 나뉜 폴더들이 저장될 위치
NUM_SPLITS = 50               # 몇 개의 폴더로 나눌지 (10개)
# ==========================================

def split_images():
    # 1. 원본 폴더 확인
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 에러: '{SOURCE_DIR}' 폴더를 찾을 수 없습니다.")
        print("스크립트 파일과 같은 위치에 'images' 폴더가 있는지 확인해주세요.")
        return

    # 2. 이미지 파일 목록 가져오기
    # (숨김 파일이나 시스템 파일 제외하고 이미지 확장자만)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tar', '.gz')
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions) or '.' in f]
    
    # 혹시 몰라 정렬 (순서대로 나누기 위해)
    all_files.sort()
    
    total_files = len(all_files)
    print(f"📂 총 파일 개수: {total_files}개")

    if total_files == 0:
        print("❌ 이동할 파일이 없습니다.")
        return

    # 3. 한 폴더당 들어갈 파일 개수 계산 (올림 처리)
    chunk_size = math.ceil(total_files / NUM_SPLITS)
    print(f"📦 한 폴더당 약 {chunk_size}개씩 분배합니다.\n")

    # 4. 폴더 생성 및 파일 이동
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    for i in range(NUM_SPLITS):
        # 폴더 이름 생성 (Hospital_01, Hospital_02 ...)
        folder_name = f"Hospital_{str(i+1).zfill(2)}"
        target_folder = os.path.join(OUTPUT_BASE_DIR, folder_name)
        
        # 폴더 생성
        os.makedirs(target_folder, exist_ok=True)

        # 자를 범위 계산
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # 리스트 슬라이싱
        batch_files = all_files[start_idx:end_idx]

        print(f"🚀 [{folder_name}] 생성 중... ({len(batch_files)}장 이동)")

        # 파일 이동 (move) -> 원본을 남기고 싶으면 shutil.copy로 변경하세요
        for filename in batch_files:
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(target_folder, filename)
            shutil.move(src_path, dst_path)

    print("\n✨ 모든 작업이 완료되었습니다!")
    print(f"📁 '{OUTPUT_BASE_DIR}' 폴더를 확인해보세요.")

if __name__ == "__main__":
    split_images()