import pytest
import unredactor

def test_main(capfd):
    unredactor.main()
    out,err = capfd.readouterr()
    assert type(out) == str

