from typing import Optional

from cinnamon_core.core.configuration import Configuration
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.callback import guard


class MyCallback(Callback):

    def on_fit_begin(self, logs=None):
        print('On fit begin')

    def on_fit_end(self, logs=None):
        print('On fit end')


@guard()
def pred(callbacks: Optional[MyCallback] = None):
    print('doing prediction')


@guard()
def fit(callbacks: Optional[MyCallback] = None):
    print('doing fitting')


def test_decorated_function(capsys):
    pred(callbacks=MyCallback(config=Configuration()))
    captured_stdout = capsys.readouterr()
    assert captured_stdout.out == "doing prediction\n"

    fit(callbacks=MyCallback(config=Configuration()))
    captured_stdout = capsys.readouterr()
    assert captured_stdout.out == "On fit begin\ndoing fitting\nOn fit end\n"
