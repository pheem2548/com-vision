from ultralytics import YOLO

model = YOLO("best.pt")


results2 = model("test/011.jpg")
results2[0].show()