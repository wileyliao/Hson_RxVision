import torch
from torchvision import transforms, models

def stage_2_main(term_data):
    cls_model = models.resnet18(pretrained=False)
    cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 2)
    cls_model.load_state_dict(torch.load(r".\model\stage2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    cls_model = cls_model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    result_map = {}
    for term, data in term_data.items():
        patch = data["patch"]
        input_tensor = transform(patch).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            output = cls_model(input_tensor)
            _, pred = torch.max(output, 1)
            face_type = pred.item()  # 0: 文字面, 1: 藥丸面

        result_map[term] = {
            "frame_path": data["frame_path"],
            "patch": patch,
            "face_type": face_type
        }

    return result_map
