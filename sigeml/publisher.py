import json
import logging
import random
import time

import paho.mqtt.client as mqtt

from constants import MODEL_TOPIC

logging.basicConfig(level=logging.INFO)


class MQTT:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = mqtt.Client()
        self.client.enable_logger(self.logger)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.process_message

        self.client.connect("mqtt-broker", 1883, 60)

        self.client.loop_start()

    def on_connect(self, **kwargs):  # pylint: disable=unused-argument
        self.client.subscribe(MODEL_TOPIC)

    def send_data(self, topic, state):
        self.client.publish(topic, state)

    def process_message(self, client, userdata, msg):  # pylint: disable=unused-argument
        try:
            msg_data = json.loads(msg.payload.decode())
            self.logger.info(f"Message: {msg_data}")
        except TypeError as e:
            self.logger.error(f"{e}")


if __name__ == "__main__":
    mq = MQTT()
    while True:
        mq.send_data(
            MODEL_TOPIC, json.dumps({"prediction": random.randint(1000, 1200)})
        )
        time.sleep(10)
