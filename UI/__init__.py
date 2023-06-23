import gradio as gr

def AppUI(predictor):
  TOP_K = 5
  def submit(inputImage):
    results = predictor(inputImage)
    # results is a dict of {model_name: [(class_name, probability), ...]}
    asMarkdown = '# Results\n\n'
    for modelName, modelResults in results.items():
      modelResults = modelResults[:TOP_K]
      asMarkdown += f'## {modelName}\n\n'
      asMarkdown += '\n'.join([
        '%d. %s: %.1f%%' % (i, className, probability * 100)
        for i, (className, probability) in enumerate(modelResults, 1)
      ])
      
      asMarkdown += '\n\n'
      continue

    return asMarkdown
  
  with gr.Blocks() as app:
    with gr.Column():
      inputImage = gr.Image(type='numpy', label='Input image', interactive=True, shape=(224, 224))
      submitButton = gr.Button(value='Submit')

    with gr.Column():
      resultsAreas = gr.Markdown()

    # Bindings
    submitButton.click(submit, inputs=[inputImage], outputs=[resultsAreas])
    pass
  return app