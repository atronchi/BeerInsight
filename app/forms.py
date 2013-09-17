from flask.ext.wtf import Form
from wtforms import TextField, DecimalField, BooleanField
from wtforms.validators import Required

class LoginForm(Form):
    openid = TextField('openid', 
        validators = [Required()]
        )
    remember_me = BooleanField('remember_me', 
        default = False
        )

class UserInputForm(Form):
    brewer = TextField('brewer',
        default='Brewer Name', 
        validators = [Required()]
        )
    beer = TextField('beer',
        default='Beer Name', 
        validators = [Required()]
        )
    rating = DecimalField('rating', places=0, default=10,
        validators = [Required()]
        )

