import os
import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34_UNet

def rle_encode(mask):
    """
    mask: numpy array, shape [H, W], values 0 or 1
    return: RLE string
    """
    pixels = mask.flatten(order='F')  # column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_inference(model, device, test_txt_path, image_dir, output_csv):
    
    if not os.path.exists(test_txt_path):
        print(f"⚠️ 找不到名單檔案: {test_txt_path}")
        return

    with open(test_txt_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]

    #Transform but NO Data Augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'encoded_mask'])

        with torch.no_grad():
            for idx, file_name in enumerate(file_names):
                img_path = os.path.join(image_dir, file_name + '.jpg')
                
                if not os.path.exists(img_path):
                    print(f"pictures not found: {img_path}！")
                    writer.writerow([file_name, ""])
                    continue

               
                original_img = Image.open(img_path).convert("RGB")
                original_size = original_img.size 

               
                img_tensor = transform(original_img).unsqueeze(0).to(device)
                output = model(img_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()

                #  256x256 to original size ， threshold(0.5)  0/1 Mask
                mask_img = Image.fromarray((prob > 0.5).astype(np.uint8))
                mask_img = mask_img.resize(original_size, Image.NEAREST)
                final_mask = np.array(mask_img)

                # RLE  to CSV
                encoded_mask = rle_encode(final_mask)
                writer.writerow([file_name, encoded_mask])

    print(f"finish {len(file_names)} to path: {output_csv}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using : {device}")

    # ==========================================
    # 統一使用「相對路徑」，確保助教可以直接執行
    # ==========================================
    image_dir = './dataset/oxford-iiit-pet/images'
    
    # UNet inference 
    print("\n--- Start  UNet inference ---")
    unet_weight_path = './saved_models/best_unet.pth' 
    if os.path.exists(unet_weight_path):
        model_unet = UNet(in_channels=3, out_channels=1).to(device)
        model_unet.load_state_dict(torch.load(unet_weight_path, map_location=device))
        model_unet.eval()
        run_inference(
            model=model_unet,
            device=device,
            test_txt_path='./dataset/test_unet.txt',
            image_dir=image_dir,
            output_csv='./predictions_best_unet/submission.csv'
        )
    else:
        print(f"❌ 找不到權重檔: {unet_weight_path}")

    # ：ResNet34_UNet inference
    print("\n--- 開始執行 ResNet34_UNet 推論 ---")
    resnet_weight_path = './saved_models/RestNet34_unet.pth' 
    if os.path.exists(resnet_weight_path):
        model_resnet = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
        model_resnet.load_state_dict(torch.load(resnet_weight_path, map_location=device))
        model_resnet.eval()
        run_inference(
            model=model_resnet,
            device=device,
            test_txt_path='./dataset/test_res_unet.txt',
            image_dir=image_dir,
            output_csv='./predictions_resnet34_unet/submission.csv'
        )
    else:
        print(f"❌ 找不到權重檔: {resnet_weight_path}")

if __name__ == '__main__':
    main()