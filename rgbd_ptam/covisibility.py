from threading import Lock

from collections import defaultdict, Counter
from itertools import chain



"""
It has many measurements and corresponding mappoints
KeyFrame is GraphKeyFrame
"""
class GraphKeyFrame(object):
    def __init__(self):
        self.id = None
        self.meas = dict() # Measurement key and value mappoint
        self.covisible = defaultdict(int) # If another frame is covisible and also count how mant times 
        self._lock = Lock()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphKeyFrame) and
            self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id   # predate

    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):
        with self._lock:
            return self.meas.keys()

    def mappoints(self):
        with self._lock:
            return self.meas.values()

    def add_measurement(self, m):
        with self._lock:
            self.meas[m] = m.mappoint

    def remove_measurement(self, m):
        with self._lock:
            try:
                del self.meas[m]
            except KeyError:
                pass

    def covisibility_keyframes(self):
        with self._lock:
            return self.covisible.copy()  # shallow copy

    def add_covisibility_keyframe(self, kf):
        with self._lock:
            self.covisible[kf] += 1


"""
Mappoint coresponds to many measurements
Mappoint is GraphMapPoint
"""
class GraphMapPoint(object):
    def __init__(self):
        self.id = None
        self.meas = dict() # Measurements keys and values keyframes
        self._lock = Lock()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMapPoint) and
            self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):
        with self._lock:
            return self.meas.keys()

    def keyframes(self):
        with self._lock:
            return self.meas.values()

    def add_measurement(self, m):
        with self._lock:
            self.meas[m] = m.keyframe

    def remove_measurement(self, m):
        with self._lock:
            try:
                del self.meas[m]
            except KeyError:
                pass

"""
Measurement is GraphMeasurement
"""
class GraphMeasurement(object):
    def __init__(self):
        self.keyframe = None
        self.mappoint = None

    @property
    def id(self): # See also meas_lookup
        return (self.keyframe.id, self.mappoint.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMeasurement) and
            self.id == rhs.id)


class CovisibilityGraph(object):
    def __init__(self, ):
        self._lock = Lock()

        self.kfs = []
        self.pts = set()

        self.kfs_set = set()
        self.meas_lookup = dict() # id -> measurement

    def keyframes(self):
        with self._lock:
            return self.kfs.copy()

    def mappoints(self):
        with self._lock:
            return self.pts.copy()

    def add_keyframe(self, kf):
        with self._lock:
            self.kfs.append(kf)
            self.kfs_set.add(kf)

    def add_mappoint(self, pt):
        with self._lock:
            self.pts.add(pt)

    def remove_mappoint(self, pt):
        with self._lock:
            try:
                for m in pt.measurements():
                    m.keyframe.remove_measurement(m)
                    del self.meas_lookup[m.id]
                self.pts.remove(pt)
            except:
                pass

    def add_measurement(self, kf, pt, meas):
        # We have a frame kf and one 3D point with measurements and a new measurement for this 3D point
        # For this keyframe we add covisible keyframes based on the meadurements of the 3D point
        with self._lock:
            if kf not in self.kfs_set or pt not in self.pts:
                return
            for m in pt.measurements():
                if m.keyframe == kf:
                    continue

                # Link keyframe kf with 3D point measurement keyframes #
                kf.add_covisibility_keyframe(m.keyframe)

                # Link 3D measurement keyframes with the kf
                m.keyframe.add_covisibility_keyframe(kf)

            # Add new measurement #
            meas.keyframe = kf
            meas.mappoint = pt
            kf.add_measurement(meas)
            pt.add_measurement(meas)

            self.meas_lookup[meas.id] = meas

    def remove_measurement(self, m):
        # Given a measurement remove it from a keyframe                 #
        # Also, remove this measurement from its corresponding mappoint #
        m.keyframe.remove_measurement(m)
        m.mappoint.remove_measurement(m)
        with self._lock:
            try:
                del self.meas_lookup[m.id]
            except:
                pass

    def has_measurement(self, *args):
        with self._lock:
            if len(args) == 2: # keyframe, mappoint
                id = (args[0].id, args[1].id)
                return id in self.meas_lookup
            else:
                raise TypeError

    def get_reference_frame(self, seedpoints):
        assert len(seedpoints) > 0

        visible = [pt.keyframes() for pt in seedpoints]
        visible = Counter(chain(*visible))

        return visible.most_common(1)[0][0]

    """
        Given 3D points get most probable frame -> reference
        Also, for the reference frame get covisibl
    """
    def get_local_map(self, seedpoints, window_size=15):
        reference = self.get_reference_frame(seedpoints)
        covisible = chain(
            reference.covisibility_keyframes().items(), [(reference, float('inf'))])
        covisible = sorted(covisible, key=lambda _:_[1], reverse=True)

        local_map = [seedpoints]
        local_keyframes = []
        for kf, n in covisible[:window_size]:
            if n < 1:
                continue

            local_map.append(kf.mappoints())
            local_keyframes.append(kf)

        local_map = list(set(chain(*local_map)))
        return local_map, local_keyframes

    """
        Typically, seedframes: [previous, reference]
    """
    def get_local_map_v2(self, seedframes, window_size=12, loop_window_size=8):
        # TODO: add similar (in appearance and location) keyframes' mappoints
        covisible = []
        for kf in set(seedframes):
            covisible.append(Counter(kf.covisibility_keyframes()))

        # Find all occurances and set to inf the seedframes #
        covisible = sum(covisible, Counter())
        for kf in set(seedframes):
            covisible[kf] = float('inf')

        local = sorted(
            covisible.items(), key=lambda _:_[1], reverse=True)

        # Get most recent frame #
        id = max([_.id for _ in covisible])

        # Get loop frames: 20 frames before the most recent one #
        loop_frames = [_ for _ in local if _[0].id < id-20]

        local = local[:window_size]
        loop_local = []
        if len(loop_frames) > 0:

            # Find covisible of loop #
            loop_covisible = sorted(
                loop_frames[0][0].covisibility_keyframes().items(),
                key=lambda _:_[1], reverse=True)

            # If not in local add #
            for kf, n in loop_covisible:
                if kf not in set([_[0] for _ in local]):
                    loop_local.append((kf, n))
                    if len(loop_local) >= loop_window_size:
                        break
 
        local = chain(local, loop_local)

        local_map = []
        local_keyframes = []
        for kf, n in local:
            if n < 1:
                continue

            local_map.append(kf.mappoints())
            local_keyframes.append(kf)

        local_map = list(set(chain(*local_map)))
        return local_map, local_keyframes
