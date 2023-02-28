import bpy
import bmesh
from mathutils import Vector
from math import floor
import random
from bpy.props import FloatVectorProperty

SPECTRAL = [
	(158,1,66),
	(213,62,79),
	(244,109,67),
	(253,174,97),
	(254,224,139),
	(255,255,191),
	(230,245,152),
	(171,221,164),
	(102,194,165),
	(50,136,189),
	(94,79,162),
]

def clamp(x, min, max):
	if (x <= min): return min 
	if (x >= max): return max

	return x

class Box:
	# min = Vector([0, 0, 0])
	# max = Vector([0, 0, 0])

	def __init__(self):
		self.min = Vector([0, 0, 0])
		self.max = Vector([0, 0, 0])

	def toCube(self):
		cubesize = max(
			self.max.x - self.min.x,
			self.max.z - self.min.z,
			self.max.y - self.min.y)

		box = Box()
		box.min = Vector([self.min.x, self.min.y, self.min.z])
		box.max = Vector([
			self.min.x + cubesize,
			self.min.y + cubesize,
			self.min.z + cubesize,
		])

		return box

	def getCenter(self):
		center = Vector([
			(self.min.x + self.max.x) / 2.0,
			(self.min.y + self.max.y) / 2.0,
			(self.min.z + self.max.z) / 2.0
		])

		return center

	def getSize(self):
		size = Vector([
			self.max.x - self.min.x,
			self.max.y - self.min.y,
			self.max.z - self.min.z
		])

		return size


class Node:

	def __init__(self):
		self.level = 0
		self.coords = []
		self.box = Box()
		self.test = 0


def getAABB(object):

	boxmin = Vector([ float("inf"),  float("inf"),  float("inf")])
	boxmax = Vector([-float("inf"), -float("inf"), -float("inf")])

	vertices = object.data.vertices
	coords = [(object.matrix_world @ v.co) for v in vertices]
	for coord in coords: 
		boxmin.x = min(boxmin.x, coord.x)
		boxmin.y = min(boxmin.y, coord.y)
		boxmin.z = min(boxmin.z, coord.z)
		boxmax.x = max(boxmax.x, coord.x)
		boxmax.y = max(boxmax.y, coord.y)
		boxmax.z = max(boxmax.z, coord.z)

	# for vertex in object.bound_box: 
	# 	v = object.matrix_world @ Vector(vertex)
	# 	boxmin.x = min(boxmin.x, v.x)
	# 	boxmin.y = min(boxmin.y, v.y)
	# 	boxmin.z = min(boxmin.z, v.z)
	# 	boxmax.x = max(boxmax.x, v.x)
	# 	boxmax.y = max(boxmax.y, v.y)
	# 	boxmax.z = max(boxmax.z, v.z)

	box = Box()
	box.min = boxmin
	box.max = boxmax
		
	return box

def drawBoundingBox(box):
	cz = (box.max.z + box.min.z) / 2
	width = box.max.x - box.min.x
	depth = box.max.y - box.min.y
	height = box.max.z - box.min.z
	boxsize = Vector([width, depth, height])
	cubesize = max(width, depth, height)

	pwidth = cubesize / 500.0
	plength = 0.5 * cubesize

	collection_main = bpy.context.scene.collection
	collection_root = bpy.data.collections['Voxelized']
	collection_bounds = bpy.data.collections.new("BoundingBox")

	collection_root.children.link(collection_bounds)
	bpy.context.scene.collection.children.link(collection_root)

	positions = [
		(box.min.x + 0 * cubesize, box.min.y + 0 * cubesize, cz),
		(box.min.x + 1 * cubesize, box.min.y + 0 * cubesize, cz),
		(box.min.x + 0 * cubesize, box.min.y + 1 * cubesize, cz), 
		(box.min.x + 1 * cubesize, box.min.y + 1 * cubesize, cz), 
		((box.min.x + box.max.x) / 2.0, box.min.y + 0 * cubesize, box.min.z + 0 * cubesize), 
		((box.min.x + box.max.x) / 2.0, box.min.y + 1 * cubesize, box.min.z + 0 * cubesize), 
		((box.min.x + box.max.x) / 2.0, box.min.y + 0 * cubesize, box.min.z + 1 * cubesize), 
		((box.min.x + box.max.x) / 2.0, box.min.y + 1 * cubesize, box.min.z + 1 * cubesize), 
		(box.min.x + 0 * cubesize, (box.min.y + box.max.y) / 2.0, box.min.z + 0 * cubesize), 
		(box.min.x + 1 * cubesize, (box.min.y + box.max.y) / 2.0, box.min.z + 0 * cubesize), 
		(box.min.x + 0 * cubesize, (box.min.y + box.max.y) / 2.0, box.min.z + 1 * cubesize), 
		(box.min.x + 1 * cubesize, (box.min.y + box.max.y) / 2.0, box.min.z + 1 * cubesize), 
	]

	scales = [
		(pwidth, pwidth, plength),
		(pwidth, pwidth, plength),
		(pwidth, pwidth, plength),
		(pwidth, pwidth, plength),
		(plength, pwidth, pwidth),
		(plength, pwidth, pwidth),
		(plength, pwidth, pwidth),
		(plength, pwidth, pwidth),
		(pwidth, plength, pwidth),
		(pwidth, plength, pwidth),
		(pwidth, plength, pwidth),
		(pwidth, plength, pwidth),
	]

	for i in range(12): 
		bpy.ops.mesh.primitive_cube_add(
			location=positions[i],
			scale=scales[i])
		object = bpy.context.active_object
		collection_bounds.objects.link(object)
		collection_main.objects.unlink(object)

	bpy.ops.object.select_all(action='DESELECT')


def voxelize(object, grid, gridsize): 

	aabb = getAABB(object).toCube()
	boxsize = aabb.max - aabb.min

	coords = [(object.matrix_world @ v.co) for v in vertices]

	for coord in coords:
		
		fx = gridsize * (coord.x - aabb.min.x) / boxsize.x
		fy = gridsize * (coord.y - aabb.min.y) / boxsize.y
		fz = gridsize * (coord.z - aabb.min.z) / boxsize.z

		ix = clamp(floor(fx), 0, gridsize - 1)
		iy = clamp(floor(fy), 0, gridsize - 1)
		iz = clamp(floor(fz), 0, gridsize - 1)

		voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize

		isTargetOctant = False
		isFinalOctant = False

		if(gridsize == 32): 
			isTargetOctant = (ix < gridsize / 2) and (iy < gridsize / 2) and (iz >= gridsize / 2)
		elif(gridsize == 64):
			isTargetOctant = (ix < gridsize / 2) and (iy < gridsize / 2) and (iz >= gridsize / 2)

			isFinalOctant = (ix < gridsize / 4) 
			isFinalOctant = isFinalOctant and (iy < gridsize / 4) 
			isFinalOctant = isFinalOctant and (iz < 3 * gridsize / 4) 
			isFinalOctant = isFinalOctant and (iz >= 2 * gridsize / 4) 

			isTargetOctant = (not isTargetOctant)

		isFinalOctant = False
		isTargetOctant = False

		if(isFinalOctant): 
			grid[voxelIndex] = 10
		elif (not isTargetOctant):
			grid[voxelIndex] = 1


def chunking(parent, object):

	depth = 3
	grids = [
		[[] for i in range(1 * 1 * 1)],
		[[] for i in range(2 * 2 * 2)],
		[[] for i in range(4 * 4 * 4)],
		[[] for i in range(8 * 8 * 8)],
	]

	aabb = getAABB(object).toCube()
	boxsize = aabb.max - aabb.min
	vertices = object.data.vertices
	coords = [(object.matrix_world @ v.co) for v in vertices]

	for i in range(len(coords)):

		coord = coords[i]
		
		gridsize = pow(2, depth)
		fx = gridsize * (coord.x - aabb.min.x) / boxsize.x
		fy = gridsize * (coord.y - aabb.min.y) / boxsize.y
		fz = gridsize * (coord.z - aabb.min.z) / boxsize.z

		ix = clamp(floor(fx), 0, gridsize - 1)
		iy = clamp(floor(fy), 0, gridsize - 1)
		iz = clamp(floor(fz), 0, gridsize - 1)

		voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize

		grids[depth][voxelIndex].append(coord)

	for level in [2]:

		gridsize = pow(2, level)

		for voxelIndex in range(gridsize * gridsize * gridsize):
			x = voxelIndex % gridsize
			y = floor((voxelIndex % (gridsize * gridsize)) / gridsize)
			z = floor(voxelIndex / (gridsize * gridsize))

			toNextLevelIndex = lambda x,y,z : x + y * (2 * gridsize) + z * (4 * gridsize * gridsize)

			i000 = toNextLevelIndex(2 * x + 0, 2 * y + 0, 2 * z + 0)
			i001 = toNextLevelIndex(2 * x + 0, 2 * y + 0, 2 * z + 1)
			i010 = toNextLevelIndex(2 * x + 0, 2 * y + 1, 2 * z + 0)
			i011 = toNextLevelIndex(2 * x + 0, 2 * y + 1, 2 * z + 1)
			i100 = toNextLevelIndex(2 * x + 1, 2 * y + 0, 2 * z + 0)
			i101 = toNextLevelIndex(2 * x + 1, 2 * y + 0, 2 * z + 1)
			i110 = toNextLevelIndex(2 * x + 1, 2 * y + 1, 2 * z + 0)
			i111 = toNextLevelIndex(2 * x + 1, 2 * y + 1, 2 * z + 1)

			count = 0
			count = count + len(grids[level + 1][i000])
			count = count + len(grids[level + 1][i001])
			count = count + len(grids[level + 1][i010])
			count = count + len(grids[level + 1][i011])
			count = count + len(grids[level + 1][i100])
			count = count + len(grids[level + 1][i101])
			count = count + len(grids[level + 1][i110])
			count = count + len(grids[level + 1][i111])

			if(count < 1500):
				grids[level][voxelIndex] = [
					*grids[level + 1][i000],
					*grids[level + 1][i001],
					*grids[level + 1][i010],
					*grids[level + 1][i011],
					*grids[level + 1][i100],
					*grids[level + 1][i101],
					*grids[level + 1][i110],
					*grids[level + 1][i111],
				]

				grids[level + 1][i000] = []
				grids[level + 1][i001] = []
				grids[level + 1][i010] = []
				grids[level + 1][i011] = []
				grids[level + 1][i100] = []
				grids[level + 1][i101] = []
				grids[level + 1][i110] = []
				grids[level + 1][i111] = []

	nodes = []

	for level in [3, 2]:
		
		gridsize = pow(2, level)
		voxels = grids[level]
		
		for voxelIndex in range(gridsize * gridsize * gridsize):

			if(len(voxels[voxelIndex]) == 0): continue

			x = voxelIndex % gridsize
			y = floor((voxelIndex % (gridsize * gridsize)) / gridsize)
			z = floor(voxelIndex / (gridsize * gridsize))
			
			mesh = bpy.data.meshes.new("chunk")
			object_mesh = bpy.data.objects.new(mesh.name, mesh)

			colorindex = random.randint(0, len(SPECTRAL) - 1)
			r = SPECTRAL[colorindex][0] / 256.0
			g = SPECTRAL[colorindex][1] / 256.0
			b = SPECTRAL[colorindex][2] / 256.0

			object_mesh["objcolor"] = (r, g, b)

			parent.objects.link(object_mesh)
			bpy.context.view_layer.objects.active = object_mesh

			vertices = []
			edges = []
			faces = []

			for coord in voxels[voxelIndex]:
				vertices.append(coord)

			mesh.from_pydata(vertices, edges, faces)

			box = Box()
			box.min.x = aabb.min.x + x * (boxsize.x / gridsize)
			box.min.y = aabb.min.y + y * (boxsize.y / gridsize)
			box.min.z = aabb.min.z + z * (boxsize.z / gridsize)
			box.max.x = box.min.x + boxsize.x / gridsize
			box.max.y = box.min.y + boxsize.y / gridsize
			box.max.z = box.min.z + boxsize.z / gridsize
			
			node = Node()
			node.level = level
			node.coords = voxels[voxelIndex]
			node.box = box
			# node.box.min.x = voxelIndex
			nodes.append(node)
			# node.test = voxelIndex

			# print(nodes[len(nodes) - 1].box.min)
			# if(len(nodes) > 3):
			# 	print(nodes[0].box.min)
			# 	print(nodes[1].box.min)
			# 	print(nodes[2].box.min)

	for i in range(50):
		node = nodes[i]

		# print(node.test)
		# print(node.box.min)
		# print(node.box.max)
		# print(node.box.min, node.box.max, node.box.getCenter())

		bpy.ops.mesh.primitive_cube_add()
		box = bpy.context.active_object
		box.location = node.box.getCenter()
		box.scale = node.box.getSize() * 0.5

		parent.objects.link(box)
		bpy.context.scene.collection.objects.unlink(box)

		# break
		# print(nodes[0].box.min)
		# print(nodes[0].box.max)



##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################




targetCollectionName = "Chunks"
gridsize = 8


print("start")

# object = bpy.context.active_object

# object["colorTest"] = (1.0, 0.0, 0.0)
# print(object["colorTest"])

object = bpy.data.objects['bunny_30k']
# aabb = getAABB(object)

if(bpy.context.active_object):
	bpy.context.active_object.select_set(False)

if(targetCollectionName in bpy.data.collections):
	collection = bpy.data.collections[targetCollectionName]
	
	for obj in collection.objects:
		bpy.data.objects.remove(obj, do_unlink=True)

	bpy.data.collections.remove(collection)

collection = bpy.data.collections.new(targetCollectionName)
bpy.context.scene.collection.children.link(collection)




chunking(collection, object)

# print("done")



