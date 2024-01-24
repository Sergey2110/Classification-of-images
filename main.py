import os
import time
import warnings
from lib.Classification import ClassificationImage

if __name__ == "__main__":
    classification = ClassificationImage()
    start = time.time()
    train_loss_log, train_acc_log, val_loss_log, val_acc_log, metrics_file_train, metrics_file_valid = classification.crossvalid(
        num_epoch=3)
    print(f'{(time.time() - start) // 60:.0f}мин {(time.time() - start) % 50:.0f}с')

    classification.plot_history(train_loss_log, val_loss_log)
    classification.plot_history(train_acc_log, val_acc_log, 'Верность(accuracy)')

    classification.watch_dataframe(metrics_file_train[:50], True)
    classification.watch_dataframe(metrics_file_valid[:50], True)

    classification.visual_filters()
    classification.visual_maps()

    # 0 - автокран
    # 1 - легковой автомобиль
    # 2 - экскаватор
    # 3 - человек
    # 4 - самосвал
    # 5 - карьерный погрузчик
    # 6 - каток
    # 7 - бульдозер

    classification.watch_img()
    classification.evaluation_model()
    classification.save_model("ImageClassifier.pt")
    classification.save_model_ONNX()
    classification.quantize_model()
