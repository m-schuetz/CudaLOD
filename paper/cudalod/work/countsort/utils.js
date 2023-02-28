

function createTestData12(n, boxSize, min, max){
    let points = [];
    let size = [
        max[0] - min[0],
        max[1] - min[1],
    ];

    for(let i = 0; i < n; i++){
        let [x, y] = [
            Math.random() * size[0] + min[0], 
            Math.random() * size[1] + min[1],
        ];

        let nx = (x - boxSize / 2) / (boxSize / 2);
        let ny = (y - boxSize / 2) / (boxSize / 2);

        // let d = Math.sqrt( (x - 0.5 * boxSize) ** 2 + (y - 0.5 * boxSize) ** 2);
        let d = Math.sqrt(nx * nx + ny * ny);

        let rad = Math.atan2(ny, nx);
        let a = Math.sin(10 * rad);

        d = d + 0.15 * a;

        if(d > 0.8){
            continue;
        }

        points.push([x, y]);
    }

    let center = [
        0.5 * (max[0] + min[0]),
        0.5 * (max[0] + min[0]),
    ];

    let pointcloud = {
        points: points,
        min: min,
        max: max,
        center: center,
    };

    return pointcloud;
}

export function testDataWrapper(n, boxSize, gridSize, cell){

    let cellSize = boxSize / gridSize;

    let min = [
        cell[0] * cellSize, 
        cell[1] * cellSize,
    ];
    let max = [
        (cell[0] + 1) * cellSize,
        (cell[1] + 1) * cellSize,
    ];

    return createTestData12(n, boxSize, min, max);
}


function mortonCode(x, y, maxLevel){
    let mc = 0;

    for(let i = 0; i < maxLevel; i++){
        mc = mc | ((x >> i) & 1) << (2 * i + 0);
        mc = mc | ((y >> i) & 1) << (2 * i + 1);
    }

    return mc;
}


export const svgns = "http://www.w3.org/2000/svg";

export function createDisplay(size){
    let svg = document.createElementNS(svgns, "svg");

    svg.setAttribute("class", "cartesian");
	svg.setAttribute("width", size);
	svg.setAttribute("height", size);

    let element = document.createElement("span");
    let label = document.createElement("span");

    document.body.appendChild(element);
    element.appendChild(svg);
    element.appendChild(label);

    return {
        element: element,
        svg: svg,
        label: label,
    };
};



export function createTestData(n, boxSize, gridSize, maxLevel){
    let order = [];
    for(let x = 0; x < gridSize; x++){
    for(let y = 0; y < gridSize; y++){

        let mc = mortonCode(x, y, maxLevel);
        order.push({mc: mc, xy: [x, y]});
    }
    }

    order.sort( (a, b) => {
        return a.mc - b.mc;
    });

    let pointclouds = [];
    for(let abc of order){
        let [x, y] = abc.xy;
        let data = testDataWrapper(n, boxSize, gridSize, [x, y]);
        pointclouds.push(data);
    }

    return pointclouds;
}


export function distance(a, b){
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dd = dx * dx + dy * dy;

    return Math.sqrt(dd);
}


// from: https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
export function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }

    return array;
}


export function createCircle(x, y, r){
	let circle = document.createElementNS(svgns, "circle");

	circle.setAttributeNS(null, "cx", x);
	circle.setAttributeNS(null, "cy", y);
	circle.setAttributeNS(null, "r", r);
	circle.setAttributeNS(null, "style", "fill: black; stroke: none");

	return circle;
	//svg.appendChild(circle);
};


export function show(pointcloud, size, target){

    let {min, max, center, points} = pointcloud;

    let svg = target ?? document.createElementNS(svgns, "svg");

    let lines = [];
    for(let a of points){

        let [x, y] = a;

        let src = `<use
            style="display:inline"
            transform="translate(${x},${y})"
            height="100%"
            width="100%"
            y="0"
            x="0"
            xlink:href="#path9890"
            inkscape:spray-origin="#path9890"
            id="use9923" />
        `;

        let circle = createCircle(x, y, size);
		svg.appendChild(circle);
    }

    return svg;
}