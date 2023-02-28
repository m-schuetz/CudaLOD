
import * as fs from 'fs';
const fsp = fs.promises;
import JSON5 from './json5.mjs';

const path = "./results_ca21_bunds.json";
// const path = "./results_retz.json";
// const path = "./results_palmyra.json";
// const path = "./results_saint_roman.json";

// returns benchmark result with median total duration from group
function getMedian(group){
	group.sort( (a, b) => a["duration_total"] - b["duration_total"]);

	let medianIndex = Math.floor(group.length / 2);

	return group[medianIndex];
}

let txtJson = (await fsp.readFile(path)).toString();
let json = JSON5.parse(txtJson);

let benchmarks = json.benchmarks;

{ // some sanity checks
	let setDevice = new Set(benchmarks.map(b => b.device))
	let setNumPoints = new Set(benchmarks.map(b => b.points))

	let abort = (msg) => {
		console.error(msg);
		process.exit(1);
	};
	
	if(setDevice.size !== 1) abort("results from different cuda devices. expecting result file to originate from a single device.");
	if(setNumPoints.size !== 1) abort("results with different point count. expected results originate from single point cloud");
}

console.log("#######################");
console.log("## BENCHMARK RESULTS ##");
console.log("#######################");

console.log(`reporting results from median time of each strategy`);
console.log(`device: ${benchmarks[0].device}`);
console.log(`#points: ${benchmarks[0].points.toLocaleString()}`);
console.log();

let groups = new Map();
let strategies = [];

// group by sampling strategy
for(let benchmark of benchmarks){
	if(!groups.has(benchmark.strategy)){
		groups.set(benchmark.strategy, []);
		strategies.push(benchmark.strategy);
	}

	groups.get(benchmark.strategy).push(benchmark);
}

for(let strategy of strategies){
	console.log(`## ${strategy}`);

	let group = groups.get(strategy);
	
	let median = getMedian(group);

	let strd_total     = median.duration_total.toFixed(1).padStart(6);
	let strd_split     = median.duration_split.toFixed(1).padStart(6);
	let strd_voxel     = median.duration_voxelize.toFixed(1).padStart(6);
	let str_throughput = median.throughput.toFixed(1).padStart(6);
	
	console.log(`duration[split]:          ${strd_split} ms`);
	console.log(`duration[voxelize]:       ${strd_voxel} ms`);
	console.log(`duration[total]:          ${strd_total} ms`);
	console.log(`throughput                ${str_throughput} M points/s`);
	console.log();
}


