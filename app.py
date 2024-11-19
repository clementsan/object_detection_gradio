# app.py

import gradio as gr
#import spaces
#import torch

from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
import io

model_pipeline = pipeline(model="facebook/detr-resnet-50")


COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


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
def detect(image):
    results = model_pipeline(image)
    print(results)

    output_figure = get_output_figure(image, results, threshold=0.9)

    buf = io.BytesIO()
    output_figure.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    output_pil_img = Image.open(buf)

    return output_pil_img


with gr.Blocks() as demo:
    gr.Markdown("# Object detection with DETR on COCO dataset")
    gr.Markdown(
        """
        This application uses a DETR (DEtection TRansformers) model to detect objects on images.
        This version was trained using the COCO dataset.
        You can load an image and see the predictions for the objects detected.
        """
    )

    gr.Interface(
        fn=detect,
        inputs=gr.Image(label="Input image", type="pil"),
        outputs=[gr.Image(label="Output prediction", type="pil")],
        examples=['samples/savanna.jpg'],
    )

demo.launch(show_error=True)