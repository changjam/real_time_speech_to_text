import torch
import gradio as gr
from method import start_recording, stop_recording, load_model

with gr.Blocks() as GUI:
    gr.HTML('''<h1 align='center'>Real Time Speech to Text</h1>''')
    with gr.Column(scale=1):
        subtitle_board = gr.Text(lines = 10, max_lines=10)
        with gr.Row():
            chunk = gr.Number(label= 'chunk', value=1024, minimum=0, info='記錄聲音的樣本區塊大小')
            fs = gr.Number(label= 'fs', value=16000, minimum=0, info='取樣頻率')
            channels = gr.Number(label= 'channels', value=1, minimum=0, info='聲道數量')
            seconds = gr.Number(label= 'seconds', value=3, minimum=0, info='多久存一次')
        with gr.Row():
            task = gr.Dropdown(['translate', 'transcribe'], label='task', value='transcribe')
            detect_language = gr.Dropdown(['zh','vi','ja','en','th','ko','my','pl','jw','nn','tr','ar','ru','ms','it','fr','id','ta','nl','km','cy','hi','es','ml','de','bo','el','sv','pt','fa','la','he','ro','da','sn','hu','fi','te','tl','bn','ur','ne','br','uk','si','yo','haw','tg','sd','gu','am','be','yi','lo','ka','uz','fo','ht','ps','tk','mt','sa','lb','mg','as','ba','tt','su','ln','ha','mk','ca','cs','no','hr','bg','lt','mi','sk','lv','sr','az','sl','kn','et','oc','eu','is','hy','mn','bs','kk','sq','sw','gl','mr','pa','so','af'], label='language', value='en')
        
        start_recording_btn = gr.Button('Start Recording', variant='primary')
        stop_recording_btn = gr.Button('Stop Recording', variant='stop')

    start_recording_btn.click(start_recording,
                                inputs= [chunk,
                                        fs,
                                        channels,
                                        seconds,
                                        task, 
                                        detect_language
                                    ],
                                outputs=subtitle_board,
                                show_progress = True
                            )
    stop_recording_btn.click(stop_recording)

if __name__ == '__main__':
    print(f'use_cuda: {torch.cuda.is_available()}')
    model_size = 'large-v3'
    load_model(model_size)
    GUI.queue().launch(share=False, inbrowser=False, server_name = '127.0.0.1', server_port=8000)
