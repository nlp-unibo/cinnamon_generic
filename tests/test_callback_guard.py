from typing import Optional

from cinnamon_core.core.configuration import Configuration
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.callback import guard


class MyCallback(Callback):

    def on_fit_begin(self, logs=None):
        print('On fit begin')

    def on_fit_end(self, logs=None):
        print('On fit end')


@guard(hookpoint='on_pred')
def pred_function(callbacks: Optional[MyCallback] = None):
    print('doing prediction')


@guard(hookpoint='on_fit')
def fit_function(callbacks: Optional[MyCallback] = None):
    print('doing fitting')


def test_decorated_function(capsys):
    pred_function(callbacks=MyCallback(config=Configuration()))
    captured_stdout = capsys.readouterr()
    assert captured_stdout.out == "doing prediction\n"

    fit_function(callbacks=MyCallback(config=Configuration()))
    captured_stdout = capsys.readouterr()
    assert captured_stdout.out == "On fit begin\ndoing fitting\nOn fit end\n"
