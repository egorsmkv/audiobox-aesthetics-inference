import shutil
from os.path import basename
from glob import glob
from zipfile import ZipFile

import sphn
import gradio as gr
from audiobox_aesthetics import infer as aes_infer

BATCH_SIZE = 10

aes_predictor = aes_infer.initialize_predictor()


def make_batches(iterable, n=1):
    lnx = len(iterable)
    for ndx in range(0, lnx, n):
        yield iterable[ndx : min(ndx + n, lnx)]


def extract_zip_and_analyze(archive_name, ce, cu, pc, pq):
    n_files = 0

    shutil.rmtree("./tmp/")

    with ZipFile(archive_name, "r") as zip_file:
        zip_file.extractall("./tmp/")

        forward_value = []
        for row in glob("./tmp/*.wav"):
            forward_value.append({"path": row})

            n_files += 1

    gr.Success(f"Unarchived {n_files} files")

    final_files = []
    for batch in make_batches(forward_value, BATCH_SIZE):
        results = aes_predictor.forward(batch)

        for idx, metrics in enumerate(results):
            filename = batch[idx]["path"]

            print(f"{filename} has these metrics: {metrics}")

            if metrics["CE"] < ce:
                print("CE is low")
                continue

            if metrics["CU"] < cu:
                print("CU is low")
                continue

            if metrics["PC"] < pc:
                print("PC is low")
                continue

            if metrics["PQ"] < pq:
                print("PQ is low")
                continue

            final_files.append(filename)

    durations = sum(sphn.durations(final_files))
    volume_mins = round(durations / 60, 4)
    volume_hrs = round(durations / 60 / 60, 4)

    gr.Success(
        f"Analyzed {n_files} files, total useful files: {len(final_files)}, volume: {volume_hrs} hours ({volume_mins} minutes)"
    )

    archive_name = "filtered_files.zip"

    with ZipFile(archive_name, "w") as zip_file:
        for filename in final_files:
            arc_name = basename(filename)
            zip_file.write(filename, arc_name)

    return archive_name


demo = gr.Interface(
    title="audiobox-aesthetics inference",
    fn=extract_zip_and_analyze,
    inputs=[
        gr.File(
            label="ZIP archive with WAV files", file_count="single", file_types=[".zip"]
        ),
        gr.Slider(label="Content Enjoyment", maximum=10, minimum=0.1, value=4),
        gr.Slider(label="Content Usefulness", maximum=10, minimum=0.1, value=4),
        gr.Slider(label="Production Complexity", maximum=10, minimum=0.1, value=1.5),
        gr.Slider(label="Production Quality", maximum=10, minimum=0.1, value=4),
    ],
    outputs="file",
    submit_btn="Inference",
)
demo.launch(share=False, server_name="0.0.0.0")
