import pytest
import pandas as pd
import unredactor as unredactor

def test_get_normalized_data():
    data = [['Gnan58','training','Sasha Mitchell','But ██████████████ stole the show with his Cody character.']]
    df = pd.DataFrame(data, columns = ['github_name', 'type', 'redacted_name', 'redacted_sentence',])
    actual = unredactor.get_normalized_data(df)
    assert type(actual) == type(df)