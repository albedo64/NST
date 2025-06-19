from flask import Flask, request, send_file, render_template
from nst_class import NST
import pathlib

app = Flask(__name__)

@app.route("/")
def render_default():
    return render_template("index.html")

@app.route('/api', methods=['POST'])
def api():
    image1_path = str(request.form.get('image1'))
    image2_path = str(request.form.get('image2'))
    iteration = request.form.get('iteration')
    if iteration is not None:
        iteration = int(iteration)
    else: iteration = 15

    nst = NST()
    output_image_path = nst.training_loop(image1_path, image2_path, iteration)
    
    img = nst.display_image(output_image_path)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run()