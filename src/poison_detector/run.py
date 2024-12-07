from poison_detector.util import load_data, DEBUG, generate_labels
from poison_detector.models.gan import train_gan

train_loader = load_data()

gen, disc = train_gan(train_loader, debug=DEBUG)
generate_labels(train_loader, disc)
