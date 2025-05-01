import os
import gi
import cv2
import numpy as np
import hailo

# GStreamer setup
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# User paths
HOME = os.environ['HOME']
HEF_PATH = os.path.join(HOME, 'heat_seeking_missle/resources/yolov11s.hef')
LABELS_JSON_PATH = os.path.join(HOME, 'heat_seeking_missle/resources/fire-labels.json')

# Tracker class
class Tracker:
    def __init__(self):
        self.tracker = None
        self.tracking = False

    def init_tracker(self, frame, bbox):
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = self.tracker.init(frame, bbox)

    def update_tracker(self, frame):
        if self.tracker is not None and self.tracking:
            success, bbox = self.tracker.update(frame)
            if success:
                return bbox
            else:
                self.tracking = False
        return None

# Frame extraction helper
def extract_frame(buffer, caps):
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None

    try:
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        format = caps.get_structure(0).get_value('format')

        if format != 'RGB':
            return None

        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        return frame
    finally:
        buffer.unmap(map_info)

# GStreamer app
class DetectionApp:
    def __init__(self):
        self.tracker = Tracker()
        self.pipeline = self.build_pipeline()
        self.frame = None

        appsink = self.pipeline.get_by_name("frame_sink")
        appsink.connect("new-sample", self.on_new_sample)

    def build_pipeline(self):
        pipeline_desc = f'''
            libcamerasrc !
            video/x-raw, width=640, height=480, framerate=30/1 !
            videoscale !
            videoconvert !
            video/x-raw, format=RGB, width=640, height=480, framerate=30/1 !
            queue leaky=2 max-size-buffers=3 !
            hailocropper so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops !
            hailonet hef-path={HEF_PATH} batch-size=1 !
            queue leaky=2 max-size-buffers=3 !
            hailofilter so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so function-name=filter_letterbox labels-file={LABELS_JSON_PATH} !
            queue leaky=2 max-size-buffers=3 !
            appsink name=frame_sink emit-signals=true sync=false
        '''
        return Gst.parse_launch(pipeline_desc)

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        frame = extract_frame(buffer, caps)
        if frame is None:
            return Gst.FlowReturn.OK

        self.frame = frame.copy()
        self.process_detections(buffer)
        self.display_frame()

        return Gst.FlowReturn.OK

    def process_detections(self, buffer):
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()

            if confidence > 0.5:
                h, w, _ = self.frame.shape
                x = int(bbox.xmin() * w)
                y = int(bbox.ymin() * h)
                width = int((bbox.xmax() - bbox.xmin()) * w)
                height = int((bbox.ymax() - bbox.ymin()) * h)

                if not self.tracker.tracking:
                    self.tracker.init_tracker(self.frame, (x, y, width, height))
                else:
                    # draw detection boxes
                    cv2.rectangle(self.frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

    def display_frame(self):
        if self.frame is not None:
            if self.tracker.tracking:
                bbox = self.tracker.update_tracker(self.frame)
                if bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Detection + Tracking", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.quit()

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        bus = self.pipeline.get_bus()
        while True:
            message = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(f"Error: {err}, {debug}")
                    break
                elif message.type == Gst.MessageType.EOS:
                    print("End of stream")
                    break

        self.quit()

    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        exit(0)

if __name__ == "__main__":
    app = DetectionApp()
    app.run()
