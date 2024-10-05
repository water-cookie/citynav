import json
from collections.abc import Callable
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from gsamllavanav.defaultpaths import OBJECTS_PATH, PROCESSED_DECRIPTIONS_PATH
from gsamllavanav.space import Point2D, Point3D


@dataclass
class ProcessedDescription:
    target: str
    landmarks: list[str]
    surroundings: list[str]

@dataclass
class CityReferObject:
    map_name: str
    id: int
    name: str
    object_type: str
    position: Point3D
    dimension: Point3D
    descriptions: list[str]
    contour: list[Point2D]
    processed_descriptions: list[ProcessedDescription] = field(default_factory=list)

    def __post_init__(self):
        # cast to NamedTuple
        self.position = Point3D(*self.position)
        self.dimension = Point3D(*self.dimension)
        self.contour = [Point2D(*p) for p in self.contour]
    
    @property
    def area(self):
        return self.contour_polygon.area
    
    @property
    def bbox_corners(self):
        x, y, z = self.position
        dx, dy, dz = self.dimension

        return [
            Point2D(x - dx/2, y + dy/2),
            Point2D(x + dx/2, y + dy/2),
            Point2D(x + dx/2, y - dy/2),
            Point2D(x - dx/2, y - dy/2),
        ]

    @property
    def contour_polygon(self):
        return Polygon(self.contour + [self.contour[0]])


ObjectID = int
MapName = str
SingleMapObjects = dict[ObjectID, CityReferObject]
MultiMapObjects = dict[MapName, SingleMapObjects]


def get_city_refer_objects(objects_path=OBJECTS_PATH, processed_description_path=PROCESSED_DECRIPTIONS_PATH) -> MultiMapObjects:
    '''returns a dictionary of CityReferObject
    
    Examples
    --------
    >>> objects = get_city_refer_objects()
    >>> objects['cambridge_block_3'][1].name
    'Merton Hall'
    '''

    # load objects
    with open(objects_path) as f:
        objects = {
            map_name: {
                int(obj_id) : CityReferObject(**obj)
                for obj_id, obj in map_objects.items()
            }
            for map_name, map_objects in json.load(f).items()
        }
    
    # add processed descriptions
    with open(processed_description_path) as f:
        processed_descs = json.load(f)
    
    for map_name, map_objects in objects.items():
        for obj_id, obj in map_objects.items():
            if map_name in processed_descs and str(obj_id) in processed_descs[map_name]:
                obj.processed_descriptions = [ProcessedDescription(**desc) for desc in processed_descs[map_name][str(obj_id)]]
    
    return objects


def get_landmarks(objects_path=OBJECTS_PATH, processed_description_path=PROCESSED_DECRIPTIONS_PATH):
    return filter_landmarks(get_city_refer_objects(objects_path, processed_description_path))


def filter_objects_by_map(objects: MultiMapObjects, map_names: list[str]) -> MultiMapObjects:
    return {
        map_name: objs
        for map_name, objs in objects.items()
        if map_name in map_names
    }


def filter_objects(objects: MultiMapObjects, condition: Callable[[CityReferObject], bool]) -> MultiMapObjects:
    return {
        map_name: {
            id: obj
            for id, obj in objs.items()
            if condition(obj)
        }
        for map_name, objs in objects.items()
    }


def filter_objects_with_description(objects: MultiMapObjects):
    return filter_objects(
        objects=objects,
        condition=lambda obj: obj.descriptions
    )


def filter_landmarks(objects: MultiMapObjects):
    return filter_objects(
        objects=objects,
        condition=lambda obj: obj.name  # only landmarks have names
    )


def extract_landmarks_from_description(description: str, landmarks: SingleMapObjects):
    '''returns a list of landmarks mentioned in `description`'''
    
    def _normalize_str(string):
        return string.lower().replace(' ', '')
    
    normalized_description = _normalize_str(description)
    mentioned_landmarks = [
        landmark
        for landmark in landmarks.values()
        if _normalize_str(landmark.name) in normalized_description
    ]
    
    return mentioned_landmarks


def remove_duplicate_landmarks_by_area(landmarks: MultiMapObjects) -> MultiMapObjects:
    
    largest_landmark: MultiMapObjects = {map_name: dict() for map_name in landmarks}
    
    for map_name, single_map_landmarks in landmarks.items():
        
        largest = largest_landmark[map_name]

        for landmark in single_map_landmarks.values():
            if landmark.name not in largest or largest[landmark.name].area < landmark.area:
                largest[landmark.name] = landmark
    
    return {
        map_name: {landmark.id: landmark for landmark in single_map_landmarks.values()}
        for map_name, single_map_landmarks in largest_landmark.items()
    }