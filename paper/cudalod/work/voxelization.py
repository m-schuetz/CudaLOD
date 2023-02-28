import bpy
import bmesh
from mathutils import Vector

from math import floor

#bpy.ops.mesh.primitive_cube_add()
#box = bpy.context.active_object
#box.location = (1, 1, 0)
#box.scale = (0.5, 0.5, 0.5)

def clamp(x, min, max):
	if (x <= min): return min 
	if (x >= max): return max

	return x

class Box:
	min = Vector([0, 0, 0])
	max = Vector([0, 0, 0])

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

	print(aabb.max)
	print(aabb.min)

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
		# isTargetOctant = True

		if(isFinalOctant): 
			grid[voxelIndex] = 10
		elif (not isTargetOctant):
			grid[voxelIndex] = 1

def drawSpheres(parent, object, grid, gridsize):

	aabb = getAABB(object).toCube()
	boxsize = aabb.max - aabb.min
	voxelsize = boxsize.x / gridsize

	coords = [(object.matrix_world @ v.co) for v in object.data.vertices]

	# collection_spheres = bpy.data.collections.new("spheres")
	collection_main = bpy.context.scene.collection
	# parent.children.link(collection_spheres)


	mesh = bpy.data.meshes.new("spheres")
	object_mesh = bpy.data.objects.new(mesh.name, mesh)
	collection_main.objects.link(object_mesh)
	bpy.context.view_layer.objects.active = object_mesh

	vertices = []
	edges = []
	faces = []

	numSpheresAdded = 0

	for coord in coords:
		
		fx = gridsize * (coord.x - aabb.min.x) / boxsize.x
		fy = gridsize * (coord.y - aabb.min.y) / boxsize.y
		fz = gridsize * (coord.z - aabb.min.z) / boxsize.z

		ix = clamp(floor(fx), 0, gridsize - 1)
		iy = clamp(floor(fy), 0, gridsize - 1)
		iz = clamp(floor(fz), 0, gridsize - 1)

		voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize

		value = grid[voxelIndex]

		if(value == 10): 

			vertices.append(coord)
			# bpy.ops.mesh.primitive_ico_sphere_add(
			# 	subdivisions = 3,
			# 	radius = 0.008,
			# 	calc_uvs = True,
			# 	enter_editmode = False,
			# 	align = 'WORLD',
			# 	location = coord,
			# 	# rotation=rotation,
			# 	# scale=scale
			# )

			# object = bpy.context.active_object
			# collection_spheres.objects.link(object)
			# collection_main.objects.unlink(object)

			# numSpheresAdded = numSpheresAdded + 1
	
	mesh.from_pydata(vertices, edges, faces)




# def drawSpheres(parent, object, grid, gridsize):

# 	aabb = getAABB(object).toCube()
# 	boxsize = aabb.max - aabb.min
# 	voxelsize = boxsize.x / gridsize

# 	coords = [(object.matrix_world @ v.co) for v in vertices]

# 	collection_spheres = bpy.data.collections.new("spheres")
# 	collection_main = bpy.context.scene.collection
# 	parent.children.link(collection_spheres)

# 	numSpheresAdded = 0

# 	for coord in coords:
		
# 		fx = gridsize * (coord.x - aabb.min.x) / boxsize.x
# 		fy = gridsize * (coord.y - aabb.min.y) / boxsize.y
# 		fz = gridsize * (coord.z - aabb.min.z) / boxsize.z

# 		ix = clamp(floor(fx), 0, gridsize - 1)
# 		iy = clamp(floor(fy), 0, gridsize - 1)
# 		iz = clamp(floor(fz), 0, gridsize - 1)

# 		voxelIndex = ix + iy * gridsize + iz * gridsize * gridsize

# 		value = grid[voxelIndex]

# 		if(value == 10): 
# 			bpy.ops.mesh.primitive_ico_sphere_add(
# 				subdivisions = 3,
# 				radius = 0.008,
# 				calc_uvs = True,
# 				enter_editmode = False,
# 				align = 'WORLD',
# 				location = coord,
# 				# rotation=rotation,
# 				# scale=scale
# 			)

# 			object = bpy.context.active_object
# 			collection_spheres.objects.link(object)
# 			collection_main.objects.unlink(object)

# 			numSpheresAdded = numSpheresAdded + 1


def drawVoxels(parent, object, grid, gridsize):

	aabb = getAABB(object).toCube()
	boxsize = aabb.max - aabb.min
	voxelsize = boxsize.x / gridsize

	collection_voxels = bpy.data.collections.new("voxels")
	collection_main = bpy.context.scene.collection
	parent.children.link(collection_voxels)

	mesh = bpy.data.meshes.new("VoxelMesh")
	object_mesh = bpy.data.objects.new(mesh.name, mesh)
	collection_voxels.objects.link(object_mesh)
	bpy.context.view_layer.objects.active = object_mesh

	vertices = []
	edges = []
	faces = []

	numVoxelsAdded = 0
	facesPerVoxel = 1

	for voxelIndex in range(gridsize * gridsize * gridsize):

		value = grid[voxelIndex]

		if(value != 1): continue

		ix = voxelIndex % gridsize
		iy = floor((voxelIndex % (gridsize * gridsize)) / gridsize)
		iz = floor(voxelIndex / (gridsize * gridsize))

		fx = ix / gridsize
		fy = iy / gridsize
		fz = iz / gridsize

		s = 0.05

		# BOTTOM
		vertices.append((
			(ix + 0.0 + s) * voxelsize + aabb.min.x,
			(iy + 0.0 + s) * voxelsize + aabb.min.y,
			(iz + 0.0 + s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 1.0 - s) * voxelsize + aabb.min.x,
			(iy + 0.0 + s) * voxelsize + aabb.min.y,
			(iz + 0.0 + s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 1.0 - s) * voxelsize + aabb.min.x,
			(iy + 1.0 - s) * voxelsize + aabb.min.y,
			(iz + 0.0 + s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 0.0 + s) * voxelsize + aabb.min.x,
			(iy + 1.0 - s) * voxelsize + aabb.min.y,
			(iz + 0.0 + s) * voxelsize + aabb.min.z,
		))

		# TOP
		vertices.append((
			(ix + 0.0 + s) * voxelsize + aabb.min.x,
			(iy + 0.0 + s) * voxelsize + aabb.min.y,
			(iz + 1.0 - s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 1.0 - s) * voxelsize + aabb.min.x,
			(iy + 0.0 + s) * voxelsize + aabb.min.y,
			(iz + 1.0 - s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 1.0 - s) * voxelsize + aabb.min.x,
			(iy + 1.0 - s) * voxelsize + aabb.min.y,
			(iz + 1.0 - s) * voxelsize + aabb.min.z,
		))

		vertices.append((
			(ix + 0.0 + s) * voxelsize + aabb.min.x,
			(iy + 1.0 - s) * voxelsize + aabb.min.y,
			(iz + 1.0 - s) * voxelsize + aabb.min.z,
		))

		# BOTTOM
		faces.append([
			8 * numVoxelsAdded + 0,
			8 * numVoxelsAdded + 1,
			8 * numVoxelsAdded + 2,
			8 * numVoxelsAdded + 3,
		])

		# TOP
		faces.append([
			8 * numVoxelsAdded + 4,
			8 * numVoxelsAdded + 5,
			8 * numVoxelsAdded + 6,
			8 * numVoxelsAdded + 7,
		])

		# SIDE 1
		faces.append([
			8 * numVoxelsAdded + 0,
			8 * numVoxelsAdded + 1,
			8 * numVoxelsAdded + 5,
			8 * numVoxelsAdded + 4,
		])

		# SIDE 1
		faces.append([
			8 * numVoxelsAdded + 2,
			8 * numVoxelsAdded + 3,
			8 * numVoxelsAdded + 7,
			8 * numVoxelsAdded + 6,
		])

		# SIDE 1
		faces.append([
			8 * numVoxelsAdded + 1,
			8 * numVoxelsAdded + 2,
			8 * numVoxelsAdded + 6,
			8 * numVoxelsAdded + 5,
		])

		faces.append([
			8 * numVoxelsAdded + 3,
			8 * numVoxelsAdded + 0,
			8 * numVoxelsAdded + 4,
			8 * numVoxelsAdded + 7,
		])

		numVoxelsAdded = numVoxelsAdded + 1

		
		# bpy.ops.mesh.primitive_cube_add(
		# 	location = pos,
		# 	scale = (voxelsize / 2.2, voxelsize / 2.2, voxelsize / 2.2))

		# voxelObject = bpy.context.active_object
		# collection_voxels.objects.link(voxelObject)
		# collection_main.objects.unlink(voxelObject)

	mesh.from_pydata(vertices, edges, faces)

	# for voxelIndex in range(gridsize * gridsize * gridsize):

	# 	value = grid[voxelIndex]

	# 	if(value == 0): continue

	# 	ix = voxelIndex % gridsize
	# 	iy = floor((voxelIndex % (gridsize * gridsize)) / gridsize)
	# 	iz = floor(voxelIndex / (gridsize * gridsize))

	# 	fx = ix / gridsize
	# 	fy = iy / gridsize
	# 	fz = iz / gridsize

	# 	pos = (
	# 		(ix + 0.5) * voxelsize + aabb.min.x,
	# 		(iy + 0.5) * voxelsize + aabb.min.y,
	# 		(iz + 0.5) * voxelsize + aabb.min.z,
	# 	)
		
	# 	bpy.ops.mesh.primitive_cube_add(
	# 		location = pos,
	# 		scale = (voxelsize / 2.2, voxelsize / 2.2, voxelsize / 2.2))

	# 	voxelObject = bpy.context.active_object
	# 	collection_voxels.objects.link(voxelObject)
	# 	collection_main.objects.unlink(voxelObject)










# targetCollectionName = "Voxelized_32x32"
targetCollectionName = "Voxelized_64x64"
gridsize = 64


print("start")

object = bpy.data.objects['bunny_30k']
# object = bpy.data.objects['bunny_148k']
vertices = object.data.vertices
coords = [(object.matrix_world @ v.co) for v in vertices]

if(bpy.context.active_object):
	bpy.context.active_object.select_set(False)

if(targetCollectionName in bpy.data.collections):
	collection = bpy.data.collections[targetCollectionName]
	
	for obj in collection.objects:
		bpy.data.objects.remove(obj, do_unlink=True)

	bpy.data.collections.remove(collection)

collection = bpy.data.collections.new(targetCollectionName)
bpy.context.scene.collection.children.link(collection)

# print(object.mesh)

aabb = getAABB(object)
#drawBoundingBox(aabb.toCube())

voxels = [0] * (gridsize * gridsize * gridsize)

voxelize(object, voxels, gridsize)
drawVoxels(collection, object, voxels, gridsize)
# drawSpheres(collection, object, voxels, gridsize)

print("done")



