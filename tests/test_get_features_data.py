import pytest
import pandas as pd

import unredactor


def test_get_features_data():
    data = [['Gnan58','training','Sasha Mitchell','But ██████████████ stole the show with his Cody character.',14,58,0.0]]
    df = pd.DataFrame(data, columns = ['github_name', 'type', 'redacted_name', 'redacted_sentence', 'name_length', 'sentence_length', 'sentiment_score'])
    x,y = unredactor.get_features_data(df,'training')
    assert y == ['Sasha Mitchell']