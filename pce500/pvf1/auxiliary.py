"""PV-F1 confirmed status controls over the shared handshake engine."""

from pce500.pi3000.auxiliary import Auxiliary as ControlEngine

CAPTURE_SAMPLE_PERIOD_CANDIDATE = 4096


class CaptureClock:
    def __init__(self):
        self.period, self.next, self.ready, self.events = None, None, False, 0

    def advance(self, active, cycle):
        if not active or self.period is None:
            self.next, self.ready = None, False
            return False
        if self.ready:
            return False
        if self.next is None:
            self.next = min(cycle + self.period, (1 << 64) - 1)
        if cycle < self.next:
            return False
        self.next, self.ready = None, True
        self.events += 1
        return True


class Auxiliary:
    def __init__(self, control=False):
        self.control = control
        self.request, self.controls = bytearray(32), bytearray(3)
        self.enabled, self.starts = True, 0
        self.engine = ControlEngine() if control else None
        self.clock = CaptureClock()
        self.sample_uploads, self.last_samples = 0, []
        self.completed_strokes = 0
        self.result_ready, self.result_irq_sent = False, False
        self.result_notification_type = 0
        self.result_pages, self.result_response = {}, None
        self.recognizer = None
        self.recognition_width, self.recognition_height = 0, 0
        self.recognition_active, self.recognition_in_stroke = None, False
        self.recognition_strokes = {}
        self.classifications, self.recognized_labels = 0, []

    def configure_recognizer(self, model):
        if not self.control:
            raise ValueError("recognizer requires auxiliary candidate")
        if self.engine.capture_active:
            raise ValueError("Leave the handwriting editor before changing the model")
        self.recognizer = model
        self.recognition_active, self.recognition_in_stroke = None, False
        self.recognition_strokes = {}
        self.classifications, self.recognized_labels = 0, []

    def recognition_upload(self, samples):
        if self.recognizer is None:
            return
        for x, y in samples:
            if x == 0x8000 and y < 4:
                self.recognition_active, self.recognition_in_stroke = y, False
            elif x == 0xFFFF and y == 0xFFFF:
                self.recognition_in_stroke = False
            elif x < 0x8000 and y < 0x8000:
                if self.recognition_active is None:
                    raise ValueError("recognizer sample without start marker")
                strokes = self.recognition_strokes.setdefault(
                    self.recognition_active, []
                )
                if not self.recognition_in_stroke:
                    strokes.append([])
                    self.recognition_in_stroke = True
                if len(strokes) > 64:
                    raise ValueError("too many recognition strokes")
                stroke = strokes[-1]
                if len(stroke) >= 16384:
                    raise ValueError("too many recognition points")
                if not stroke or stroke[-1] != [float(x), float(y)]:
                    stroke.append([float(x), float(y)])
            else:
                raise ValueError(f"unrecovered recognition marker{x:04X}/{y:04X}")

    def recognition_finish(self):
        if self.recognizer is None:
            return
        index = self.recognition_active
        if index is None:
            raise ValueError("recognizer finish without character")
        self.recognition_active, self.recognition_in_stroke = None, False
        if index not in self.recognition_strokes:
            raise ValueError("recognizer finish without ink")
        strokes = self.recognition_strokes.pop(index)
        labels = self.recognizer.classify(
            strokes,
            self.recognition_width,
            self.recognition_height,
            len(self.recognizer.classes),
        )
        records, self.recognized_labels = [], []
        for label, _ in labels:
            if len(label) != 1:
                continue
            if 0x21 <= ord(label) <= 0x7E:
                label = chr(ord(label) + 0xFEE0)
            try:
                encoded = label.encode("cp932")
            except UnicodeEncodeError:
                continue
            if len(encoded) == 2:
                records.append(encoded + b"\0")
                self.recognized_labels.append(label)
            if len(records) == 10:
                break
        self.classifications += 1
        self.supply_candidate_result_fixture(index, records)

    def supply_candidate_result_fixture(self, index, candidates):
        """Opaque controller replay, without interpreting or recognizing strokes."""
        if not 0 <= index < 4 or len(candidates) > 10:
            raise ValueError("PV-F1 result requires index 0..3 and at most 10 records")
        page = [bytes(record) for record in candidates]
        if any(len(record) != 3 for record in page):
            raise ValueError("PV-F1 candidate records must be three bytes")
        if not self.control:
            raise ValueError(
                "controller result fixture requires the auxiliary candidate"
            )
        if self.result_ready or index in self.result_pages:
            raise ValueError("PV-F1 result fixture would replace an unconsumed result")
        self.result_pages[index] = page
        self.result_ready, self.result_irq_sent = True, False
        self.result_notification_type = 1

    def supply_type2_notification_fixture(self):
        """Replay an opaque type-2 notification, without inventing its trigger."""
        if not self.control:
            raise ValueError("controller notification requires the auxiliary candidate")
        if self.result_ready:
            raise ValueError(
                "PV-F1 notification would replace an unacknowledged notification"
            )
        self.result_ready, self.result_irq_sent = True, False
        self.result_notification_type = 2

    def take_result_irq_edge(self):
        if (
            self.control
            and self.enabled
            and self.controls[0] & 0x40
            and self.result_ready
            and not self.result_irq_sent
        ):
            self.result_irq_sent = True
            return True
        return False

    def configure_sample_period(self, period):
        if period is not None and (not isinstance(period, int) or period <= 0):
            raise ValueError("PV-F1 capture sample period must be positive")
        if not self.control:
            raise ValueError("capture clock requires the auxiliary control candidate")
        self.clock.period, self.clock.next, self.clock.ready = period, None, False

    def snapshot(self):
        result = {
            "control_candidate": self.control,
            "controls": list(self.controls),
            "enabled": self.enabled,
        }
        if not self.engine:
            result["starts"] = self.starts
        else:
            result.update(
                capture_active=self.engine.capture_active,
                accepted_controls=list(self.engine.accepted),
                accepted_count=self.engine.accepted_count,
                sample_period_candidate=self.clock.period,
                sample_ready=self.clock.ready,
                sample_events=self.clock.events,
                sample_uploads=self.sample_uploads,
                completed_strokes=self.completed_strokes,
                last_uploaded_samples=self.last_samples,
                result_fixture_ready=self.result_ready,
                result_fixture_indices=sorted(self.result_pages),
                recognizer_loaded=self.recognizer is not None,
                recognition_canvas=[self.recognition_width, self.recognition_height],
                classifications=self.classifications,
                recognized_labels=self.recognized_labels,
            )
        return result

    def peek(self, address):
        if address == 0x8A00:
            status = self.engine.status() if self.engine else 0
            return (status & ~2) | (2 if self.enabled else 0)
        if address == 0x8A04:
            return self.controls[0]
        if address == 0x8A05 and self.control:
            return (0x80 if self.clock.ready else 0) | (
                0x40 if self.result_ready else 0
            )
        if address == 0x8A09 and self.control:
            return self.result_notification_type
        if (
            0x8A41 <= address <= 0x8A5F
            and self.engine
            and self.engine.status() & 0x40
            and self.result_response is not None
        ):
            offset = address - 0x8A41
            return (
                self.result_response[offset]
                if offset < len(self.result_response)
                else None
            )
        if address in (0x8A05, 0x8A09):
            return 0
        if address == 0x8A40 and self.engine:
            return self.engine.response(0)
        return None

    def advance(self, cycle):
        if self.engine:
            self.engine.advance(cycle)
            return self.clock.advance(
                self.enabled
                and self.engine.capture_active
                and self.controls[0] & 0x80 != 0,
                cycle,
            )
        return False

    def write(self, address, value, cycle):
        if address == 0x8A01 and value in (2, 0xFD):
            model = self.recognizer
            period = self.clock.period
            self.__init__(self.control)
            if self.control:
                self.configure_sample_period(period)
                self.configure_recognizer(model)
            self.enabled = value == 2
        elif 0x8A20 <= address <= 0x8A3F:
            offset = address - 0x8A20
            self.request[offset] = value
            if self.engine:
                self.engine.mark_written(offset)
        elif 0x8A06 <= address <= 0x8A08:
            self.controls[address - 0x8A06] = value
            if address == 0x8A07 and value & 0x80 == 0:
                self.clock.ready, self.clock.next = False, None
            if address == 0x8A08 and value & 0x40 == 0:
                self.result_ready, self.result_irq_sent = False, False
                self.result_notification_type = 0
        elif address == 0x8A02 and value == 0x80:
            if self.engine:
                command = int.from_bytes(self.request[:2], "little")
                if command == 0x02FC:
                    index = self.request[2]
                    if index not in self.result_pages:
                        raise ValueError(
                            f"PV-F1 query index {index} has no controller result fixture"
                        )
                    page = self.result_pages[index]
                    self.engine.start_packet(self.request, cycle, 3, None, True)
                    self.result_response = bytes([len(page)]) + b"".join(page)
                    del self.result_pages[index]
                    return
                if command == 0x00FC:
                    self.engine.start_packet(self.request, cycle, 2, None, False)
                    self.completed_strokes += 1
                    self.recognition_finish()
                    return
                count = self.request[1]
                if self.request[0] == 0xFF and 1 <= count <= 7:
                    self.engine.start_packet(
                        self.request, cycle, 2 + 4 * count, None, False
                    )
                    self.sample_uploads += 1
                    self.last_samples = [
                        [
                            int.from_bytes(self.request[i : i + 2], "little"),
                            int.from_bytes(self.request[i + 2 : i + 4], "little"),
                        ]
                        for i in range(2, 2 + 4 * count, 4)
                    ]
                    self.recognition_upload(self.last_samples)
                    return
                if command not in (
                    0x03FC,
                    0x05FE,
                    0x00FE,
                    0x01FE,
                    0x02FE,
                    0x30FB,
                    0x00F9,
                    0x01F9,
                    0x02F9,
                    0x04FE,
                    0x03FE,
                ):
                    raise ValueError(
                        f"unrecovered PV-F1 auxiliary command {command:04X}"
                    )
                self.engine.start(self.request, cycle)
                if command == 0x03FE:
                    self.recognition_width = int.from_bytes(self.request[2:4], "little")
                    self.recognition_height = int.from_bytes(
                        self.request[4:6], "little"
                    )
                    self.recognition_active, self.recognition_in_stroke = None, False
                    self.recognition_strokes.clear()
            else:
                self.starts += 1
        elif address == 0x8A03 and value == 0xBF:
            if self.engine:
                self.engine.acknowledge(value)
                self.result_response = None
        else:
            raise ValueError(
                f"unrecovered PV-F1 auxiliary write {address:05X}={value:02X}"
            )
