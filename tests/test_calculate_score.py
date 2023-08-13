import pytest

import unredactor


def test_calculate_score():
    sen = "If you liked the Richard Chamberlain version of the Bourne Identity then you will like this too...███████████ does this one brilliantly, you can't help but wonder if he is really out there...I reckon he and the other main cast members probably had nightmares for weeks after doing this movie as it's so intense."
    actual = unredactor.calculate_score(sen)
    assert actual == 0.32