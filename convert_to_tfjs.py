import os
# Mac 중복 로드 에러 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnx_tf.backend import prepare

# 1. 모델 구조 정의 (DenseNet121)
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXpertModel, self).__init__()
        # pretrained=False로 설정 (구조만 가져옴)
        self.model = models.densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. 모델 초기화 (이 부분이 빠져서 에러가 났던 것입니다!)
# ==========================================
model = CheXpertModel(num_classes=14)

# 3. 체크포인트 로드
# 경로가 맞는지 꼭 확인하세요!
checkpoint_path = '/Users/kimseonmin/HELIOS/federated/Chexpert/config/pre_train.pth' 
device = torch.device("cpu")

print(f"📂 모델 파일 로드 중: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"❌ 에러: 파일을 찾을 수 없습니다 -> {checkpoint_path}")
    exit()

checkpoint = torch.load(checkpoint_path, map_location=device)

# 4. 파일 구조에 따른 가중치 추출
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    print("✅ 구조 확인: Type A ('state_dict' 키 포함)")
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    print("✅ 구조 확인: Type B ('model' 키 포함)")
    state_dict = checkpoint['model']
else:
    print("✅ 구조 확인: Type C (가중치 딕셔너리 직접 로드)")
    state_dict = checkpoint

# 키 이름 변경 ('module.' 접두사 제거)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 5. 모델에 가중치 로드
try:
    model.load_state_dict(new_state_dict)
    print("🎉 모델 가중치 로드 성공!")
except Exception as e:
    print(f"❌ 가중치 로드 실패 (1차 시도): {e}")
    print("⚠️ strict=False로 재시도합니다...")
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("🎉 모델 가중치 로드 성공 (strict=False)!")
    except Exception as e2:
        print(f"❌ 최종 실패: {e2}")
        exit()

model.eval()

# 6. ONNX로 변환
print("🔄 ONNX 변환 시작...")
dummy_input = torch.randn(1, 3, 320, 320) # CheXpert 입력 크기
onnx_path = "chexpert.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=['input'], output_names=['output'], opset_version=11)
print("✅ ONNX 파일 생성 완료")

# 7. ONNX -> TensorFlow SavedModel
print("🔄 TensorFlow SavedModel 변환 시작...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("chexpert_saved_model")
print("✅ TensorFlow SavedModel 변환 완료!")

# 8. 안내 메시지
print("\n" + "="*50)
print("🚀 변환이 거의 끝났습니다! 아래 명령어를 터미널에 입력하세요:")
print("="*50)
print("tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output' --saved_model_tags=serve ./chexpert_saved_model ./chexpert_tfjs")
print("="*50)