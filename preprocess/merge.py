import os
import shutil
from pathlib import Path


def get_max_frame_number(frames_dir):
    """frames 폴더에서 가장 큰 프레임 번호를 찾습니다."""
    if not os.path.exists(frames_dir):
        return -1
    
    max_num = -1
    for filename in os.listdir(frames_dir):
        if filename.endswith('.png'):
            try:
                num = int(os.path.splitext(filename)[0])
                max_num = max(max_num, num)
            except ValueError:
                continue
    return max_num


def copy_frames_with_offset(src_frames_dir, dst_frames_dir, offset):
    """frames 폴더의 파일들을 오프셋을 적용하여 복사합니다."""
    if not os.path.exists(src_frames_dir):
        print(f"  경고: {src_frames_dir} 폴더가 존재하지 않습니다.")
        return 0
    
    os.makedirs(dst_frames_dir, exist_ok=True)
    copied_count = 0
    
    for filename in sorted(os.listdir(src_frames_dir)):
        if filename.endswith('.png'):
            try:
                frame_num = int(os.path.splitext(filename)[0])
                new_frame_num = frame_num + offset
                new_filename = f"{new_frame_num}.png"
                
                src_path = os.path.join(src_frames_dir, filename)
                dst_path = os.path.join(dst_frames_dir, new_filename)
                
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except ValueError:
                # 숫자가 아닌 파일명은 그대로 복사
                src_path = os.path.join(src_frames_dir, filename)
                dst_path = os.path.join(dst_frames_dir, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
    
    return copied_count


def copy_mano_params_with_offset(src_mano_dir, dst_mano_dir, offset):
    """mano_params 폴더의 내용을 오프셋을 적용하여 복사합니다."""
    if not os.path.exists(src_mano_dir):
        print(f"  경고: {src_mano_dir} 폴더가 존재하지 않습니다.")
        return
    
    os.makedirs(dst_mano_dir, exist_ok=True)
    
    # mano_params 내의 각 하위 폴더/파일 처리
    for item in os.listdir(src_mano_dir):
        src_item_path = os.path.join(src_mano_dir, item)
        dst_item_path = os.path.join(dst_mano_dir, item)
        
        if os.path.isdir(src_item_path):
            # 디렉토리인 경우: bboxes, params, meshes, renders 등
            os.makedirs(dst_item_path, exist_ok=True)
            
            for filename in sorted(os.listdir(src_item_path)):
                src_file_path = os.path.join(src_item_path, filename)
                
                if os.path.isdir(src_file_path):
                    # 중첩된 디렉토리인 경우 (예: debug_persp_frames/0/)
                    # 디렉토리명이 숫자인 경우 오프셋 적용
                    try:
                        dir_num = int(filename)
                        new_dir_num = dir_num + offset
                        new_dirname = str(new_dir_num)
                        dst_file_path = os.path.join(dst_item_path, new_dirname)
                    except ValueError:
                        # 숫자가 아닌 디렉토리명은 그대로 사용
                        dst_file_path = os.path.join(dst_item_path, filename)
                    
                    if os.path.exists(dst_file_path):
                        # 이미 존재하면 내용만 복사
                        for nested_file in os.listdir(src_file_path):
                            src_nested = os.path.join(src_file_path, nested_file)
                            dst_nested = os.path.join(dst_file_path, nested_file)
                            if os.path.isdir(src_nested):
                                shutil.copytree(src_nested, dst_nested, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src_nested, dst_nested)
                    else:
                        shutil.copytree(src_file_path, dst_file_path, dirs_exist_ok=True)
                else:
                    # 파일인 경우: 숫자로 시작하는 파일명에 오프셋 적용
                    try:
                        # 파일명에서 숫자 추출 시도
                        base_name, ext = os.path.splitext(filename)
                        
                        # 숫자로 시작하는 경우 (예: "0.json", "123.obj")
                        if base_name.isdigit():
                            new_num = int(base_name) + offset
                            new_filename = f"{new_num}{ext}"
                            dst_file_path = os.path.join(dst_item_path, new_filename)
                        # 숫자_접미사 형식인 경우 (예: "0_sil.png")
                        elif '_' in base_name:
                            parts = base_name.split('_', 1)
                            if parts[0].isdigit():
                                new_num = int(parts[0]) + offset
                                new_filename = f"{new_num}_{parts[1]}{ext}"
                                dst_file_path = os.path.join(dst_item_path, new_filename)
                            else:
                                dst_file_path = os.path.join(dst_item_path, filename)
                        else:
                            dst_file_path = os.path.join(dst_item_path, filename)
                        
                        shutil.copy2(src_file_path, dst_file_path)
                    except Exception as e:
                        # 오류 발생 시 원본 파일명으로 복사
                        dst_file_path = os.path.join(dst_item_path, filename)
                        shutil.copy2(src_file_path, dst_file_path)
        
        elif os.path.isfile(src_item_path):
            # 최상위 파일인 경우 (예: K_persp.json, K_persp_frames.csv)
            # 첫 번째 폴더의 파일만 복사 (나머지는 덮어쓰기 방지)
            if not os.path.exists(dst_item_path):
                shutil.copy2(src_item_path, dst_item_path)


def merge_subject_folders(base_dir, output_dir):
    """subject_example의 하위 폴더들을 하나로 병합합니다."""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    output_frames_dir = output_path / "frames"
    output_mano_dir = output_path / "mano_params"
    
    # 하위 폴더 찾기 (숫자로 된 폴더들)
    subfolders = sorted([d for d in base_path.iterdir() 
                        if d.is_dir() and d.name.isdigit()])

    if not subfolders:
        print(f"경고: {base_dir}에 숫자로 된 하위 폴더가 없습니다.")
        return
    
    print(f"병합할 폴더: {[f.name for f in subfolders]}")
    
    frame_offset = 0

    for i, subfolder in enumerate(subfolders):
        if i+1 not in [1, 3, 5, 8]:
            continue
        print(f"\n처리 중: {subfolder.name}/")
        
        # 현재 폴더의 최대 프레임 번호 확인
        src_frames = subfolder / "frames"
        max_frame = -1
        if src_frames.exists():
            max_frame = get_max_frame_number(str(src_frames))
        
        # frames 폴더 처리
        if src_frames.exists():
            copied = copy_frames_with_offset(str(src_frames), str(output_frames_dir), frame_offset)
            print(f"  frames: {copied}개 파일 복사 완료 (오프셋: {frame_offset})")
        else:
            print(f"  경고: {src_frames} 폴더가 존재하지 않습니다.")
        
        # mano_params 폴더 처리 (frames와 동일한 오프셋 사용)
        src_mano = subfolder / "mano_params"
        if src_mano.exists():
            copy_mano_params_with_offset(str(src_mano), str(output_mano_dir), frame_offset)
            print(f"  mano_params: 복사 완료 (오프셋: {frame_offset})")
        else:
            print(f"  경고: {src_mano} 폴더가 존재하지 않습니다.")
        
        # 다음 폴더를 위한 오프셋 업데이트
        # 현재 폴더의 원본 최대 프레임 번호(max_frame)에 현재 오프셋을 더한 값 + 1이 다음 오프셋
        if max_frame >= 0:
            frame_offset = frame_offset + max_frame + 1  # 다음 번호부터 시작
    
    print(f"\n병합 완료! 결과는 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    # 설정
    base_directory = "data/subject_1"
    output_directory = "data/subject_1_merged1358"
    
    merge_subject_folders(base_directory, output_directory)

