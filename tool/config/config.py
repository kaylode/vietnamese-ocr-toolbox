import yaml

class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            try:
                return self.__dict__[attr]
            except KeyError:
                return None

    def __str__(self):
        print("##########   CONFIGURATION INFO   ##########")
        pretty(self._attr)
        return '\n'
        
def pretty(d, indent=0):
  for key, value in d.items():
    print('    ' * indent + str(key) + ':', end='')
    if isinstance(value, dict):
      print()
      pretty(value, indent+1)
    else:
      print('\t' * (indent+1) + str(value))