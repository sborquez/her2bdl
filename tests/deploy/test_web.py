import os
from nose.tools import assert_equals, raises, timed, eq_, ok_
from nose import with_setup

NOSE_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NOSE_DATA_PATH = f"{NOSE_CURRENT_DIR}/test_image.jpg"
NOSE_DATA_MODEL = 'flowers'

# ADD RUN AND STOP SERVER WITH MODELE SETUPS


def test_web_ok():
    from deploy.web.test import test_ok
    test_ok()
    
def test_web_predict():
    from deploy.web.test import test_predict_default
    test_predict_default(image_path=NOSE_DATA_PATH, model=NOSE_DATA_MODEL)