
class FakeObject(object):
  def __init__(self, data):
    for name, value in data.items():
      setattr(self, name.replace(' ', '_'), value)
      continue
    return
