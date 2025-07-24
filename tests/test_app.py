import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client

# Assurez vous que la forme de sortie du vectorizer est cohérente avec la forme
#  d'entrée du classifieur (Logistic regression)

def test_vectorizer_shape():
    from app import vectorizer, model
    assert vectorizer.get_feature_names_out().shape[0] == model.n_features_in_
     