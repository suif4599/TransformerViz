from PyQt6.QtGui import QValidator

class FloatValidator(QValidator):
    def validate(self, a0, a1):
        input_str = a0
        pos = a1
        if not input_str:
            return (QValidator.State.Intermediate, input_str or "", pos)
        if '-' in input_str:
            return (QValidator.State.Invalid, input_str, pos)
        if input_str.count('.') > 1:
            return (QValidator.State.Invalid, input_str, pos)
        if input_str.endswith('.') and input_str.count('.') == 1:
            parts = input_str.split('.')
            if parts[0] or (len(parts) > 1 and parts[1]):
                return (QValidator.State.Acceptable, input_str, pos)
            else:
                return (QValidator.State.Intermediate, input_str, pos)
        try:
            float(input_str)
            return (QValidator.State.Acceptable, input_str, pos)
        except ValueError:
            return (QValidator.State.Invalid, input_str, pos)

class PositiveIntValidator(QValidator):
    def validate(self, a0, a1):
        input_str = a0
        pos = a1
        if not input_str:
            return (QValidator.State.Intermediate, input_str or "", pos)
        if input_str.startswith('0') and len(input_str) > 1:
            return (QValidator.State.Invalid, input_str, pos)
        if not input_str.isdigit():
            return (QValidator.State.Invalid, input_str, pos)
        if int(input_str) == 0:
            return (QValidator.State.Invalid, input_str, pos)
        return (QValidator.State.Acceptable, input_str, pos)