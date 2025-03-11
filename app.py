import gradio as gr
import pandas as pd
import os
import plotly.express as px
from torchvision import transforms
import torch
import data
from train import BaseNetwork


def extract_params(model_filename: str):
    name = model_filename.removeprefix('mnist_').removesuffix('.pt')
    fields = name.split('_')
    hls = f"{[int(i) for i in fields[0].removeprefix('hls').split('-')]}"
    epochs = int(fields[1].removeprefix('e'))
    lr = float(fields[2].removeprefix('lr'))
    bs = int(fields[3].removeprefix('bs'))
    vs = float(fields[4].removeprefix('vs'))
    opt = fields[5]
    norm = fields[6].removeprefix('norm')
    return {
        'hidden_layer_sizes': hls,
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': bs,
        'val_split': vs,
        'optimizer': opt,
        'normalize': norm
    }

def read_loss_df(model_dir: str, model_filename: str):
    loss_df = pd.read_csv(os.path.join(model_dir, model_filename))
    params = extract_params(model_filename)
    for k, v in params.items():
        loss_df[k] = v
    return loss_df

class ModelStore:
    def __init__(self, model_dir: str = "models", load_models: bool = True, data_dir: str = "data"):
        self.model_dir = model_dir
        files = os.listdir(model_dir)
        model_files = [f for f in files if f.endswith(".pt")]

        self.params_df = pd.DataFrame([extract_params(f) for f in model_files])
        self.loss_df = pd.concat([read_loss_df(model_dir, f) for f in files if f.endswith("loss.csv")])
        self.perf_df = pd.concat([pd.read_csv(os.path.join(model_dir, f)) for f in files if f.endswith("performance.csv")])

        self.loss_df['hidden_layer_sizes'] = self.loss_df['hidden_layer_sizes'].astype(str)
        self.perf_df['hidden_layer_sizes'] = self.perf_df['hidden_layer_sizes'].astype(str)
        self.loss_df['normalize'] = self.loss_df['normalize'].astype(str)
        self.perf_df['normalize'] = self.perf_df['normalize'].astype(str)

        # Have access to test data for making predictions
        self.test_data = iter(data.MNISTData(batch_size=1, normalize=False, val_split=0.2, data_dir=data_dir).test_loader)

    def make_loss_curve(self, hls, epochs, lr, bs, vs, opt, norm):
        # hls_comp_val = "[" + ",".join(x for x in hls.split('-')) + "]"
        loss_df = self.loss_df[
            (self.loss_df['hidden_layer_sizes'] == hls)
            & (self.loss_df['epochs'] == epochs)
            & (self.loss_df['learning_rate'] == lr)
            & (self.loss_df['batch_size'] == bs)
            & (self.loss_df['val_split'] == vs)
            & (self.loss_df['optimizer'] == opt)
            & (self.loss_df['normalize'] == norm)
        ]
        loss_df["index"] = loss_df.apply(lambda x: vs / (1 - vs) * x["index"] if x["type"] == "train" else x["index"], axis=1)
        return px.line(loss_df, x="index", y="loss", color="type", title=f"Loss Curve for Provided Parameters")

    def subset_perf_by_free_column(self, free_column, hls, epochs, lr, bs, vs, opt, norm):
        mask = self.perf_df['hidden_layer_sizes'].apply(lambda x: True)
        if free_column != 'hls':
            mask = mask & (self.perf_df['hidden_layer_sizes'] == hls)
        if free_column != 'epochs':
            mask = mask & (self.perf_df['epochs'] == epochs)
        if free_column != 'learning_rate':
            mask = mask & (self.perf_df['learning_rate'] == lr)
        if free_column != 'batch_size':
            mask = mask & (self.perf_df['batch_size'] == bs)
        if free_column != 'val_split':
            mask = mask & (self.perf_df['val_split'] == vs)
        if free_column != 'optimizer':
            mask = mask & (self.perf_df['optimizer'] == opt)
        if free_column != 'normalize':
            mask = mask & (self.perf_df['normalize'] == norm)

        return self.perf_df[mask]

    def make_accuracy_plot(self, free_column, hls, epochs, lr, bs, vs, opt, norm):
        sub_df = self.subset_perf_by_free_column(free_column, hls, epochs, lr, bs, vs, opt, norm)
        return px.bar(sub_df.sort_values(by=free_column), x=free_column, y="accuracy", title=f"Accuracy by {free_column} for Provided Parameters")

    def make_runtime_plot(self, free_column, hls, epochs, lr, bs, vs, opt, norm):
        sub_df = self.subset_perf_by_free_column(free_column, hls, epochs, lr, bs, vs, opt, norm)
        return px.bar(sub_df.sort_values(by=free_column), x=free_column, y="runtime", title=f"Runtime by {free_column} for Provided Parameters")

    def get_random_image(self):
        loader = self.test_data
        x, y = next(loader)
        scaling_factor = 10
        x = transforms.Resize(size=(28 * scaling_factor, 28 * scaling_factor))(x)
        return x.squeeze().numpy(), y.squeeze().numpy(), None

    def make_prediction(self, image, hls, epochs, lr, bs, wv, opt, norm):
        hls_ints = [int(i) for i in hls.removeprefix('[').removesuffix(']').split(',')]
        hls_str = "-".join(str(i) for i in hls_ints)
        model_name = f"mnist_hls{hls_str}_e{epochs}_lr{lr}_bs{bs}_vs{wv}_{opt}_norm{norm}.pt"
        model_path = os.path.join(self.model_dir, model_name)

        # model = NeuralNetwork()
        model = BaseNetwork(hidden_layer_sizes=hls_ints)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        # Preprocess image
        image = transforms.ToTensor()(image)
        image = transforms.Resize((28, 28))(image)
        if norm == "True":
            image = transforms.Normalize((0.1307,), (0.3081,))(image)

        model.eval()
        with torch.no_grad():
            return model(image).argmax().item()

class App:
    def __init__(self, model_store: ModelStore, server_port: int = 7860):
        self.model_store = model_store

        hls_values = sorted(list(self.model_store.params_df['hidden_layer_sizes'].unique()))
        epochs_values = sorted(list(self.model_store.params_df['epochs'].unique()))
        lr_values = sorted(list(self.model_store.params_df['learning_rate'].unique()))
        bs_values = sorted(list(self.model_store.params_df['batch_size'].unique()))
        vs_values = sorted(list(self.model_store.params_df['val_split'].unique()))
        opt_values = sorted(list(self.model_store.params_df['optimizer'].unique()))
        norm_values = sorted(list(self.model_store.params_df['normalize'].unique()))

        with gr.Blocks() as demo:
            with gr.Tab(label="Performance Summary"):
                with gr.Row():
                    hls_opts = gr.Radio(choices=hls_values, value=hls_values[0], label="Hidden Layer Sizes")
                    epoch_opts = gr.Radio(choices=epochs_values, value=epochs_values[0], label="Epochs")
                    lr_opts = gr.Radio(choices=lr_values, value=lr_values[0], label="Learning Rate")
                    batch_opts = gr.Radio(choices=bs_values, value=bs_values[0], label="Batch Size")
                    val_opts = gr.Radio(choices=vs_values, value=vs_values[0], label="Validation Split")
                    opt_opts = gr.Radio(choices=opt_values, value=opt_values[0], label="Optimizer")
                    norm_opts = gr.Radio(choices=norm_values, value=norm_values[0], label="Normalize")
                with gr.Row():
                    free_col = gr.Radio(choices=['hls', 'epochs', 'learning_rate', 'batch_size', 'val_split', 'optimizer', 'normalize'], value='epochs', label="Free Column")
                with gr.Row():
                    with gr.Column():
                        acc_plot = gr.Plot(label="Accuracy Plot", value=self.model_store.make_accuracy_plot('epochs', hls_values[0], epochs_values[0], lr_values[0], bs_values[0], vs_values[0], opt_values[0], norm_values[0]))
                    with gr.Column():
                        rt_plot = gr.Plot(label="Runtime Plot", value=self.model_store.make_runtime_plot('epochs', hls_values[0], epochs_values[0], lr_values[0], bs_values[0], vs_values[0], opt_values[0], norm_values[0]))
                with gr.Row():
                    loss_plot = gr.Plot(label="Loss Curve", value=self.model_store.make_loss_curve(hls_values[0], epochs_values[0], lr_values[0], bs_values[0], vs_values[0], opt_values[0], norm_values[0]))
                for opts in [hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts]:
                    opts.change(self.model_store.make_accuracy_plot, inputs=[free_col, hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=acc_plot)
                    opts.change(self.model_store.make_runtime_plot, inputs=[free_col, hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=rt_plot)
                    opts.change(self.model_store.make_loss_curve, inputs=[hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=loss_plot)

                free_col.change(self.model_store.make_accuracy_plot, inputs=[free_col, hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=acc_plot)
                free_col.change(self.model_store.make_runtime_plot, inputs=[free_col, hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=rt_plot)

            with gr.Tab(label="Model Explorer"):
                with gr.Row():
                    hls_opts = gr.Radio(choices=hls_values, value=hls_values[0], label="Hidden Layer Sizes")
                    epoch_opts = gr.Radio(choices=epochs_values, value=epochs_values[0], label="Epochs")
                    lr_opts = gr.Radio(choices=lr_values, value=lr_values[0], label="Learning Rate")
                    batch_opts = gr.Radio(choices=bs_values, value=bs_values[0], label="Batch Size")
                    val_opts = gr.Radio(choices=vs_values, value=vs_values[0], label="Validation Split")
                    opt_opts = gr.Radio(choices=opt_values, value=opt_values[0], label="Optimizer")
                    norm_opts = gr.Radio(choices=norm_values, value=norm_values[0], label="Normalize")
                with gr.Row():
                    image_button = gr.Button(value="Generate Random Image")
                    pred_button = gr.Button(value="Make Prediction")
                with gr.Row():
                    prediction = gr.Textbox(label="Prediction")
                    real_label = gr.Textbox(label="Real Label")
                with gr.Row():
                    # Make an image to generate random from dataset using a button
                    # Show prediction in a textbox below
                    image = gr.Image(label="Random Image", width=600, height=600)

                image_button.click(self.model_store.get_random_image, inputs=[], outputs=[image, real_label, prediction])
                pred_button.click(self.model_store.make_prediction, inputs=[image, hls_opts, epoch_opts, lr_opts, batch_opts, val_opts, opt_opts, norm_opts], outputs=prediction)



        demo.launch(server_port=server_port)
