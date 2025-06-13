import re
import sys


class Flags:
    def __init__(self, *args, **kwargs):
        from .configs import Config

        self._config = Config(*args, **kwargs)

    def parse(self, argv=None, known_only=False, help_exists=None):
        if help_exists is None:
            help_exists = not known_only
        if argv is None:
            argv = sys.argv[1:]
        if "--help" in argv:
            print("\nHelp:")
            lines = str(self._config).split("\n")[2:]
            print("\n".join("--" + re.sub(r"[:,\[\]]", "", x) for x in lines))
            help_exists and sys.exit()
        parsed = {}
        remaining = []
        key = None
        vals = None
        for arg in argv:
            if arg.startswith("--"):
                if key:
                    self._submit_entry(key, vals, parsed, remaining)
                if "=" in arg:
                    key, val = arg.split("=", 1)
                    vals = [val]
                else:
                    key, vals = arg, []
            else:
                if key:
                    vals.append(arg)
                else:
                    remaining.append(arg)
        self._submit_entry(key, vals, parsed, remaining)
        parsed = self._config.update(parsed)
        if known_only:
            return parsed, remaining
        else:
            for flag in remaining:
                if flag.startswith("--"):
                    raise ValueError(f"Flag '{flag}' did not match any config keys.")
            assert not remaining, remaining
            return parsed

    def _submit_entry(self, key, vals, parsed, remaining):
        if not key and not vals:
            return
        if not key:
            vals = ", ".join(f"'{x}'" for x in vals)
            raise ValueError(f"Values {vals} were not preceeded by any flag.")
        name = key[len("--") :]
        if "=" in name:
            remaining.extend([key] + vals)
            return
        if self._config.IS_PATTERN.match(name):
            pattern = re.compile(name)
            keys = {k for k in self._config.flat if pattern.match(k)}
        elif name in self._config:
            keys = [name]
        else:
            keys = []
        if not keys:
            remaining.extend([key] + vals)
            return
        if not vals:
            raise ValueError(f"Flag '{key}' was not followed by any values.")
        for key in keys:
            parsed[key] = self._parse_flag_value(self._config[key], vals, key)

    def _parse_flag_value(self, default, value, key):
        value = value if isinstance(value, (tuple, list)) else (value,)
        if isinstance(default, (tuple, list)):
            if len(value) == 1 and "," in value[0]:
                value = value[0].split(",")
            return tuple(self._parse_flag_value(default[0], [x], key) for x in value)
        assert len(value) == 1, value
        value = str(value[0])
        if default is None:
            return value
        if isinstance(default, bool):
            try:
                return bool(["False", "True"].index(value))
            except ValueError:
                message = f"Expected bool but got '{value}' for key '{key}'."
                raise TypeError(message)
        if isinstance(default, int):
            value = float(value)  # Allow scientific notation for integers.
            if float(int(value)) != value:
                message = f"Expected int but got float '{value}' for key '{key}'."
                raise TypeError(message)
            return int(value)
        return type(default)(value)
