import pytest
import pandas as pd
import unredactor as unredactor

def test_get_feature():
    data = [['Gnan58','training','Sasha Mitchell','But ██████████████ stole the show with his Cody character.']]
    data1 = [['Gnan58','training','Sasha Mitchell','But ██████████████ stole the show with his Cody character.',14,58,0.0]]
    df = pd.DataFrame(data, columns = ['github_name', 'type', 'redacted_name', 'redacted_sentence',])
    df1 = pd.DataFrame(data1, columns = ['github_name', 'type', 'redacted_name', 'redacted_sentence', 'name_length', 'sentence_length', 'sentiment_score'])
    actual = unredactor.feature(df)
    assert type(actual) == type(df1)