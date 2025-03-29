from PyQt5.QtGui import QValidator

class FloatValidator(QValidator):
    def validate(self, input_str, pos):
        if not input_str:
            return (QValidator.Intermediate, input_str, pos)
        if '-' in input_str:
            return (QValidator.Invalid, input_str, pos)
        if input_str.count('.') > 1:
            return (QValidator.Invalid, input_str, pos)
        if input_str.endswith('.') and input_str.count('.') == 1:
            parts = input_str.split('.')
            if parts[0] or (len(parts) > 1 and parts[1]):
                return (QValidator.Acceptable, input_str, pos)
            else:
                return (QValidator.Intermediate, input_str, pos)
        try:
            float(input_str)
            return (QValidator.Acceptable, input_str, pos)
        except ValueError:
            return (QValidator.Invalid, input_str, pos)

class PositiveIntValidator(QValidator):
    def validate(self, input_str, pos):
        if not input_str:
            return (QValidator.Intermediate, input_str, pos)
        if input_str.startswith('0') and len(input_str) > 1:
            return (QValidator.Invalid, input_str, pos)
        if not input_str.isdigit():
            return (QValidator.Invalid, input_str, pos)
        if int(input_str) == 0:
            return (QValidator.Invalid, input_str, pos)
        return (QValidator.Acceptable, input_str, pos)