echo Before V4l setting
v4l2-ctl --list-ctrls

        #constant setting fitur camera
        v4l2-ctl --set-ctrl focus_auto=0
        v4l2-ctl --set-ctrl white_balance_temperature_auto=0
        v4l2-ctl --set-ctrl white_balance_temperature=5000
        v4l2-ctl --set-ctrl exposure_auto_priority=0
        v4l2-ctl --set-ctrl exposure_auto=1
        v4l2-ctl --set-ctrl exposure_absolute=250

        v4l2-ctl --set-ctrl brightness=128
        v4l2-ctl --set-ctrl contrast=128
        v4l2-ctl --set-ctrl saturation=128
        v4l2-ctl --set-ctrl sharpness=0

echo After value
v4l2-ctl --list-ctrls
