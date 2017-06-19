import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
#sys.path.insert(0,'/home/kevin/projects/CrowdCounting/caffe-mcnn-1/python')
import caffe

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        ima = exifutil.open_oriented_im(filename)
        image = Image.open(filename)
        image = np.array(image, dtype=np.float32)
        image = image[:, :, ::-1]
        image -= np.array((104.00698793, 116.66876762, 122.67891434))
        image = image.transpose((2, 0, 1))

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    #image_pil = Image.fromarray((result[1]).astype('uint8'))
    dstname_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
    dstname = os.path.join(UPLOAD_FOLDER, dstname_)
    dst_ = result[1]/255.0
    dst_ = dst_.flatten().reshape((dst_.shape[1],dst_.shape[2]))
    plt.imsave(dstname, dst_)
    dst = exifutil.open_oriented_im(dstname)

    '''
    dst = Image.fromarray(result[1].astype('uint8'))
    dst.save()
    dst = result[1]/255.0
    dst = dst.flatten().reshape((dst.shape[1],dst.shape[2]))
    #dst = dst.transpose((1, 2, 0))
    '''


    return flask.render_template(
        'index.html',has_result=True, result=result,
        imagesrc=embed_image_html(ima),
        imagedst=embed_image_html(dst)
    )

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    #image_pil = image_pil.resize((256, 256))
    #image_pil = Image.fromarray((image).astype('uint8'))
    image_pil = image_pil.resize((image.shape[0], image.shape[1]))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    #data = string_buf.getvalue().replace('\n', '')
    return 'data:image/png;base64,' + data
    #return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/mcnn/ShanghaiTech/Part_A/MCNN/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/web_demo/MCNN_iter_3367800.caffemodel'.format(REPO_DIRNAME)),
    }
    '''
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    '''
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    #default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
        #self.net = caffe.Classifier(
        #    model_def_file, pretrained_model_file,
        #    image_dims=(image_dim, image_dim),
        #     channel_swap=(2, 1, 0)
        #)


    def classify_image(self, image):
        try:
            starttime = time.time()
            #scores = self.net.predict([image], oversample=False).flatten()
            self.net.blobs['data'].reshape(1, *image.shape)
            #self.net.blobs['data'].data[...] = image
            image = image[np.newaxis, ...]
            out = self.net.forward(**{'data':image})
            out = self.net.blobs['convo'].data[0]

            endtime = time.time()
            num = out[0].sum()

            #num = scores.sum()
            #indices = (-scores).argsort()[:5]
            #predictions = self.labels[indices]


            return (True, out, num, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
