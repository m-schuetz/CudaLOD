
// convert a point cloud from potree2 format to workgroup-render format

import * as fs from "fs";
const fsp = fs.promises;

// const fs = require('fs');
// const fsp = fs.promises;

// let path = "F:/pointclouds/benchmark/retz/morton.las_converted";
// let outpath = "F:/temp/wgtest/retz";
// let path = "E:/dev/pointclouds/benchmark/lifeboat/morton.las_converted";
// let outpath = "E:/temp/wgtest";
// let outpath = "F:/temp/wgtest/lion";


let root_encode = null;
let root_decode = null;

class BitWriter{

	constructor(numBits){

		this.buffer = Buffer.alloc(Math.ceil(numBits / 8));
		this.bitsWritten = 0;

	}

	write(value, numBits){

		// if(this.bitsWritten < 20 * 3){
		// 	console.log({value, numBits});
		// }

		let pos_word_0 = this.bitsWritten % 32;
		let len_word_0 = Math.min(pos_word_0 + numBits, 32) - pos_word_0;
		let len_word_1 = numBits - len_word_0;
		let word_index_0 = Math.floor(this.bitsWritten / 32) * 4;

		// write word 0
		let cap_word_0 = (1 << (len_word_0)) - 1;
		let mask_0 = ((value & cap_word_0) << pos_word_0) >>> 0;
		let word_0 = this.buffer.readUint32LE(word_index_0);
		let new_word_0 = (word_0 | mask_0) >>> 0;
		this.buffer.writeUint32LE(new_word_0, word_index_0);

		// write word 1
		if(len_word_1 > 0){
			let mask_1 = value >> len_word_0;
			let word_1 = this.buffer.readUint32LE(word_index_0 + 4);
			let new_word_1 = word_1 | mask_1;
			this.buffer.writeUint32LE(new_word_1, word_index_0 + 4);
		}

		this.bitsWritten += numBits;
	}

	print(){
		let bytes = [];
		for(let i = 0; i < this.buffer.byteLength; i++){
			let byte = this.buffer.readUint8(i);
			let str = byte.toString(2).padStart(8, "0");
			str = str.split("").reverse().join("");

			bytes.push(str);
		}

		console.log(bytes.join("_"));
	}

};

class BitReader{

	constructor(buffer){
		this.buffer = buffer;
	}

	read(bitPosition, numBits){
		let pos_word_0 = bitPosition % 32;
		let len_word_0 = Math.min(pos_word_0 + numBits, 32) - pos_word_0;
		let len_word_1 = numBits - len_word_0;
		let word_index_0 = Math.floor(bitPosition / 32) * 4;

		let cap_word_0 = (1 << (len_word_0)) - 1;
		let word_0 = (this.buffer.readUint32LE(word_index_0) >> pos_word_0) & cap_word_0;
		let word_1 = 0;
		
		if(len_word_1 > 0){
			let cap_word_1 = (1 << (len_word_1)) - 1;
			word_1 = this.buffer.readUint32LE(word_index_0 + 4) & cap_word_1;
		}

		let value = word_0 | word_1;

		return value;
	}

};


class Voxel{

	constructor(){
		this.x = 0;
		this.y = 0;
		this.z = 0;
		this.r = 0;
		this.g = 0;
		this.b = 0;
	}

};

class Node{

	constructor(){
		this.index = null;
		this.level = 0;
		this.children = new Array(8).fill(null);
		this.count = 0;
		this.counter = 0;
		this.color = [0, 0, 0];
	}

	add(voxel){

		this.count++;

		if(this.level >= 8){
			this.color = [voxel.r, voxel.g, voxel.b];
			return;
		}

		let bx = (voxel.x >> (7 - this.level)) & 1;
		let by = (voxel.y >> (7 - this.level)) & 1;
		let bz = (voxel.z >> (7 - this.level)) & 1;

		let childIndex = bx | (by << 1) | (bz << 2);

		if(this.children[childIndex] === null){
			let child = new Node();
			child.index = childIndex;
			child.level = this.level + 1;
			// child.color = voxel.color;
			
			this.children[childIndex] = child;
		}

		this.children[childIndex].add(voxel);
	}

	traverse(callback){
		callback(this);

		for(let child of this.children){
			if(!child) continue;

			child.traverse(callback);
		}
	}

	flatten(){

		let list = [];

		this.traverse((node) => {
			list.push(node);
		});

		return list;
	}

	childMask(){
		let childMask = 0;
		
		for(let i = 0; i < 8; i++){
			if(this.children[i] != null){
				childMask = childMask | (1 << i);
			}
		}

		return childMask;
	}

	compare(node){

		let a = this.flatten();
		let b = node.flatten();

		if(a.length !== b.length){
			console.log(`unequal lengths: ${a.length} vs. ${b.length}`);
		}

		for(let i = 0; i < Math.min(a.length, b.length); i++){

			let node_a = a[i];
			let node_b = b[i];

			let childMask_a = node_a.childMask();
			let childMask_b = node_b.childMask();
			if(childMask_a !== childMask_b){

				console.log(`unequal childMask: ${childMask_a} vs. ${childMask_b}`);
				console.log(`i: ${i}`);
				console.log(`node.level: ${node.level}`);
				console.log(`node.index: ${node.index}`);

				return;
			}

		}

		console.log("all ok");

	}

	print(first, count){

		let numProcessed = 0;

		this.traverse(node => {

			if(numProcessed > first && numProcessed < first + count){
				// console.log("    ".repeat(node.level) + `[${node.level}, ${node.index}]: ${node.childMask()}`);
				if(node.color){
					console.log("    ".repeat(node.level) + `[${node.counter}][${node.level}, ${node.index}]: ${node.childMask()}, color: ${node.color}`);
				}else{
					console.log("    ".repeat(node.level) + `[${node.counter}][${node.level}, ${node.index}]: ${node.childMask()}`);
				}
			}

			numProcessed++;
		});

	}

};

function toImage(node){

	let colors = [];

	node.traverse(node => {
		if(node.level === 8){
			colors.push(node.color);
		}
	});


	let roundedLog4 = Math.ceil(Math.log(colors.length) / Math.log(4));
	let imageSize = Math.sqrt(4 ** roundedLog4);

	let buffer = Buffer.alloc(18 + imageSize * imageSize * 3);
	// let view = new DataView(buffer);
	buffer.writeUint8(2, 2);
	buffer.writeUint16LE(imageSize, 12);
	buffer.writeUint16LE(imageSize, 14);
	buffer.writeUint8(24, 16);
	buffer.writeUint8(32, 17);

	let toMortonCoordinate = (i) => {
		let X = 0;
		let Y = 0;

		for(let bitIndex = 0; bitIndex < 10; bitIndex++){
			X = X | (((i >>> (2 * bitIndex + 0)) & 1) << bitIndex);
			Y = Y | (((i >>> (2 * bitIndex + 1)) & 1) << bitIndex);
		}

		return [X, Y];
	};

	console.log({imageSize});

	for(let i = 0; i < colors.length; i++){
		let color = colors[i];
		let imageCoord = toMortonCoordinate(i);
		let bufferCoord = imageCoord[0] + imageSize * imageCoord[1];

		buffer.writeUint8(color[0], 18 + 3 * bufferCoord + 0);
		buffer.writeUint8(color[1], 18 + 3 * bufferCoord + 1);
		buffer.writeUint8(color[2], 18 + 3 * bufferCoord + 2);
	}

	fs.writeFile("F:/temp/voxels.tga", buffer, () => {
		console.log("done");
	});

}

async function run_encode(path, targetPath){
	
	let txt = (await fsp.readFile(path)).toString();

	let lines = txt.split("\n");

	let voxels = [];
	// let buffer = Buffer.alloc(lines.length * 3);

	let root = new Node();
	root.index = null;
	root.level = 0;

	root_encode = root;

	lines.forEach( (line, i) => {

		let tokens = line.split(",");
		if(tokens.length !== 6) return;

		let numberTokens = tokens.map(token => parseInt(token));

		let xyz = [numberTokens[0], numberTokens[1], numberTokens[2]];
		let rgb = [numberTokens[3], numberTokens[4], numberTokens[5]];

		let voxel = new Voxel();
		voxel.x = numberTokens[0];
		voxel.y = numberTokens[1];
		voxel.z = numberTokens[2];
		voxel.r = numberTokens[3];
		voxel.g = numberTokens[4];
		voxel.b = numberTokens[5];

		// let voxel = {xyz, rgb};

		voxels.push(voxel);
		root.add(voxel);
	});

	console.log("#voxels: " + voxels.length);


	let numNodes = 0;

	root.traverse(node => {

		// if(!Number.isInteger(node.index)) return;
		if(node.level === 8) return;

		numNodes++;
	});

	let numProcessed = 0;
	// let bitwriter = new BitWriter(numNodes * 3 + 32);
	let buffer = Buffer.alloc(numNodes);
	root.traverse(node => {

		if(node.level === 8) return;

		let childMask = 0;
		for(let i = 0; i < 8; i++){
			if(node.children[i] != null){
				childMask = childMask | (1 << i);
			}
		}

		buffer.writeUint8(childMask, numProcessed);

		numProcessed++;

	});

	fs.writeFile(targetPath, buffer, () => {
		console.log("done");
	});

	
	
}

async function run_decode(path){

	let buffer = await fsp.readFile(path);

	// let bitreader = new BitReader(buffer);

	// let numNodes = bitreader.buffer.byteLength;
	let numNodes = buffer.byteLength;
	let childMasks = new Uint8Array(numNodes);

	let strDbg = "";
	for(let i = 0; i < numNodes; i++){
		// let value = bitreader.read(8 * i, 8);
		let value = buffer.readUint8(i);

		childMasks[i] = value;

		if(i < 20){
			strDbg += value + ", ";
		}
	}
	// console.log(strDbg);

	class StackElement{
		constructor(childMask, node){
			this.childMask = childMask;
			this.childrenProcessed = 0;
			this.node = node;
		}
	};

	let root = new Node();
	root.index = null;
	root.level = 0;

	let stack = [new StackElement(childMasks[0], root)];
	let pos = 1;

	let numProcessed = 0;

	while(stack.length > 0){

		let current = stack.at(-1);

		let childIndex = current.childrenProcessed;

		// if(pos === 6) debugger;

		if(current.node.level >= 7){
			stack.pop();
			continue;
		}

		let childExists = ((current.childMask >> childIndex) & 1) === 1;
		if(childExists){
			let childs_childMask = childMasks[pos];
			pos++;

			let childNode = new Node();
			childNode.index = childIndex;
			childNode.level = current.node.level + 1;
			current.node.children[childIndex] = childNode;
			childNode.counter = numProcessed;
			numProcessed++;

			let childElement = new StackElement(childs_childMask, childNode);

			if(childNode.level < 8){
				stack.push(childElement);
			}
			
			if(childNode.level === 7){
				for(let i = 0; i < 8; i++){
					let childExists = ((childs_childMask >> i) & 1) === 1;
					if(childExists){
						let node = new Node();
						node.level = childNode.level + 1;
						node.index = i;
						childNode.children[i] = node;
						node.counter = numProcessed;
						numProcessed++;
					}
				}
			}
		}

		current.childrenProcessed++;
		
		if(current.childrenProcessed >= 8){
			// stack.pop();
			stack = stack.filter(e => e !== current);
		}

	}


	root_decode = root;

	root_encode.compare(root_decode);

	console.log("========================");
	root_encode.print(75, 10);
	console.log("========================");
	// root_decode.print(75, 10);
	// console.log("========================");


}

let path = "F:/temp/voxels.csv";
let path_encoded = "F:/temp/voxels.bin";

async function run(){

	await run_encode(path, path_encoded);
	await run_decode(path_encoded);

	toImage(root_encode);

}

run();

// {
// 	let bitwriter = new BitWriter(128);

// 	// bitwriter.write(0b101, 3);
// 	// bitwriter.write(0b000, 3);
// 	// bitwriter.write(0b111, 3);
// 	// bitwriter.write(0b100000001, 9);
// 	// bitwriter.write(0b100000001, 9);
// 	// bitwriter.write(0b100000001, 9);
// 	// bitwriter.write(0b100000001, 9);

// 	bitwriter.write(0b1000000_00000000_00000000_00000011, 31);
// 	bitwriter.write(0b1000000_00000000_00000000_00000011, 31);
	
// 	bitwriter.print();

// }

console.log("done");