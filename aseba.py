import dbus
import dbus.mainloop.glib
from time import sleep
import os

os.system("(asebamedulla ser:name=Thymio-II &)")

dbus.mainloop.glib.threads_init()
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
bus = dbus.SessionBus()

# Create Aseba network
# if network is None:
network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                         dbus_interface='ch.epfl.mobots.AsebaNetwork')
network.SetVariable("thymio-II", "motor.left.target", [-100])
network.SetVariable("thymio-II", "motor.right.target", [100])
sleep(4)
network.SetVariable("thymio-II", "motor.left.target", [100])
network.SetVariable("thymio-II", "motor.right.target", [-100])
sleep(4)
network.SetVariable("thymio-II", "motor.left.target", [0])
network.SetVariable("thymio-II", "motor.right.target", [0])

os.system("pkill -f asebamedulla")
