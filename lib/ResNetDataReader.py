import numpy as np
import os
import time

from onnxruntime import InferenceSession
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image


class ResNetDataReader(CalibrationDataReader):
    """
    Класс для получения данных для калибровки
    """
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        session = InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        self.nhwc_data_list = self._preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def _preprocess_images(self, images_folder: str, height: int, width: int, size_limit: int = 0):
        unconcatenated_batch_data = []
        image_names = os.listdir(images_folder)

        if 0 < size_limit <= len(image_names):
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names

        for image_name in batch_filenames:
            if image_name.split('.')[1] == 'csv':
                continue
            image_filepath = images_folder + "/" + image_name
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = np.float32(pillow_img) - np.array(
                [123.68, 116.78, 103.94], dtype=np.float32
            )
            nhwc_data = np.expand_dims(input_data, axis=0)
            nchw_data = nhwc_data.transpose(0, 3, 1, 2)
            unconcatenated_batch_data.append(nchw_data)

        batch_data = np.concatenate(
            np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
        )

        return batch_data

    def benchmark(self, model_path: str):
        total = 0.0
        runs = 10
        session = InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_data = np.zeros((1, 3, 224, 224), np.float32)
        _ = session.run([], {input_name: input_data})

        for i in range(runs):
            start = time.perf_counter()
            _ = session.run([], {input_name: input_data})
            end = (time.perf_counter() - start) * 1000
            total += end
            # print(f"{end:.2f}ms")

        total /= runs
        size = os.path.getsize(model_path)
        print(f"Среднее время: {total:.2f}мс \t Размер (MB): {size / 1e6}")

    def rewind(self):
        self.enum_data = None
