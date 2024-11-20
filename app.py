# app.py

import gradio as gr
#import spaces
#import torch

from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import os

list_models = ["facebook/detr-resnet-50"]
list_models_simple = [os.path.basename(model) for model in list_models]

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def load_pipeline(model):
    model_pipeline = pipeline(model=model)
    return model_pipeline


def get_output_figure(pil_img, results, threshold):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())
        if score > threshold:
            c = COLORS[hash(label) % len(COLORS)]
            ax.add_patch(
                plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color=c, linewidth=3)
            )
            text = f"{label}: {score:0.2f}"
            ax.text(box[0], box[1], text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")

    return plt.gcf()


#@spaces.GPU
def detect(image, model_id, threshold=0.9):
    print("model:", list_models[model_id])

    model_pipeline = load_pipeline(list_models[model_id])
    results = model_pipeline(image)
    print(results)

    output_figure = get_output_figure(image, results, threshold=threshold)

    buf = io.BytesIO()
    output_figure.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    output_pil_img = Image.open(buf)

    return output_pil_img


def demo():
    with gr.Blocks(theme="base") as demo:
        gr.Markdown("# Object detection on COCO dataset")
        gr.Markdown(
            """
            This application uses transformer-based models to detect objects on images.
            This version was trained using the COCO dataset.
            You can load an image and see the predictions for the objects detected.
            """
        )

        with gr.Row():
            model_id = gr.Radio(list_models, \
                               label="Detection models", value=list_models[0], type="index", info="Choose your detection model")
        with gr.Row():
            threshold = gr.Slider(0, 1.0, value=0.9, label='Detection threshold', info="Choose your detection threshold")

        with gr.Row():
            input_image = gr.Image(label="Input image", type="pil")
            output_image = gr.Image(label="Output image", type="pil")

        with gr.Row():
            submit_btn = gr.Button("Submit")
            clear_button = gr.ClearButton()

        gr.Examples(['samples/savanna.jpg'], inputs=input_image)

        submit_btn.click(fn=detect, inputs=[input_image, model_id, threshold], outputs=[output_image])
        clear_button.click(lambda: [None, None], \
                        inputs=None, \
                        outputs=[input_image, output_image], \
                        queue=False)

    demo.queue().launch(debug=True)

if __name__ == "__main__":
    demo()