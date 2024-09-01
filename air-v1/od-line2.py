import cv2
import time
import json
from ultralytics import YOLO

def get_resolution_input():
    options = {
        "1": (640, 480),
        "2": (800, 600),
        "3": (1024, 768)
    }
    user_input = input("Escolha a resolução (1: 640x480, 2: 800x600, 3: 1024x768): ").strip() or "1"
    return options.get(user_input, (640, 480))

def save_frame_and_metadata(frame, box, class_name, save_path, direction):
    timestamp = int(time.time())
    frame_filename = f"{save_path}/{direction}_crossing_{timestamp}.jpg"
    metadata_filename = f"{save_path}/{direction}_crossing_{timestamp}.json"
    
    cv2.imwrite(frame_filename, frame)
    
    metadata = {
        "timestamp": timestamp,
        "class": class_name,
        "bounding_box": {
            "x1": int(box[0]),
            "y1": int(box[1]),
            "x2": int(box[2]),
            "y2": int(box[3])
        },
        "direction": direction
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

resolution = get_resolution_input()

orientation = input("Linhas orientadas na horizontal ou vertical? (h/v) [Padrão: h]: ").strip() or "h"
min_confidence = float(input("Insira a taxa de confiança mínima (0 a 100) [Padrão 50]: ").strip() or "50") / 100
save_path = input("Insira o caminho para salvar os vídeos e metadados [Padrão: 'D:\\Lab\\Output_AIR\\Step_1']: ").strip() or "D:\\Lab\\Output_AIR\\Step_1"

if orientation == "h":
    center_y = resolution[1] // 2
    line_1 = center_y - 10
    line_2 = center_y + 10
else:
    center_x = resolution[0] // 2
    line_1 = center_x - 10
    line_2 = center_x + 10

cap = cv2.VideoCapture(0)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

model = YOLO("yolov8n.pt")

logo_path = r'D:\GoogleDrive\TRABALHO\RADIX\Clientes\PAX-Aeroportos\Git\AIR\air-1\air-v1\images\Logo.jpg'
logo = cv2.imread(logo_path)

if logo is None:
    print(f"Erro: Não foi possível carregar o logo do caminho {logo_path}.")
else:
    logo = cv2.resize(logo, (100, 100), interpolation=cv2.INTER_AREA)

logo_height, logo_width, _ = logo.shape if logo is not None else (0, 0, 0)

def overlay_logo(img, logo, x_offset=10, y_offset=10):
    if logo is not None:
        y1, y2 = y_offset, y_offset + logo_height
        x1, x2 = x_offset, x_offset + logo_width
        img[y1:y2, x1:x2] = logo

crossed_line_1 = False
crossed_line_2 = False
line_color_1 = (0, 255, 0)  # Verde
line_color_2 = (0, 0, 255)  # Vermelho

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img)

    if orientation == "h":
        cv2.line(img, (0, line_1), (resolution[0], line_1), line_color_1, 2)
        cv2.line(img, (0, line_2), (resolution[0], line_2), line_color_2, 2)
    else:
        cv2.line(img, (line_1, 0), (line_1, resolution[1]), line_color_1, 2)
        cv2.line(img, (line_2, 0), (line_2, resolution[1]), line_color_2, 2)

    overlay_logo(img, logo)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            confidence = box.conf[0]
            if class_name in ["airplane", "helicopter"] and confidence >= min_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                if orientation == "h":
                    center_pos = (y1 + y2) // 2
                    if center_pos > line_1 and not crossed_line_1:
                        crossed_line_1 = True
                    if center_pos > line_2 and crossed_line_1 and not crossed_line_2:
                        crossed_line_2 = True
                        line_color_1 = (255, 255, 255)  # Branco
                        line_color_2 = (255, 255, 255)  # Branco
                        save_frame_and_metadata(img, (x1, y1, x2, y2), class_name, save_path, "h")
                else:
                    center_pos = (x1 + x2) // 2
                    if center_pos > line_1 and not crossed_line_1:
                        crossed_line_1 = True
                    if center_pos > line_2 and crossed_line_1 and not crossed_line_2:
                        crossed_line_2 = True
                        line_color_1 = (255, 255, 255)  # Branco
                        line_color_2 = (255, 255, 255)  # Branco
                        save_frame_and_metadata(img, (x1, y1, x2, y2), class_name, save_path, "v")

                if crossed_line_1 and crossed_line_2:
                    crossed_line_1 = False
                    crossed_line_2 = False

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Teste commit