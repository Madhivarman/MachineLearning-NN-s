//initial model definition
const model = tf.sequential();
model.add(tf.layers.dense({units:256, inputShape: [8]})); //1x8
model.add(tf.layers.dense({units:512, inputShape: [256], activation:"sigmoid"}));
model.add(tf.layers.dense({units:256, inputShape:[512], activation:"sigmoid"}));
model.add(tf.layers.dense({units:3,inputShape:[256]})); //returns 1x3

const lr = 0.001;
const optimizer = tf.train.adam(lr);
model.compile({loss:'meanSquaredError',optimizer:optimizer});

//requestAnimationFrame invocation
//it functions lot like a setTimeout but more lot useful in optimized way
//if tab isn't active it will stop making calls until its becomes active again

var animate = window.requestAnimationFrame || window.webkitRequestAnimationFrame ||
				window.mozRequestAnimationFrame ||
				function(callback) {window.setTimeout(callback, 1000/60)};


//setup a canvas and get its 2D content
var canvas = document.createElement('canvas');
var width = 600;
var height = 600;
canvas.width = width;
canvas.height = height;
var context = canvas.getContext('2d')

//setup canvas function when windows get reload
window.onload =  function(){
	document.body.appendChild(canvas);
	animate(step);
};

// first we will update the objects: The Players paddle, AI paddle, and ball
// next step is to render those objects
//last step is requestAnimationFrame to call the step function again 
var update = function() {
	//player update
	player.update();
	//ai
	if(computer.ai_plays){
		move = ai.predict_move();
		computer.ai_update(move);
	}else{
		computer.update(ball);
	}
	//now update the ball
	//bounce the ball bounce back and forth
	ball.update(player.paddle,computer.paddle);
	//save ai data for further learning
	ai.save_data(player.paddle,computer.paddle,ball);
};

var render = function(){
	context.fillStyle = "#D3D3D3";
	context.fillRect(0,0,width,height);
	player.render(); //render the player
	computer.render(); //render the AI computer 
	ball.render(); //render the ball
};

// lets add ball to the canvas
// define paddle
function Paddle(x,y, width, height){
	this.x = x;
	this.y = y;
	this.width = width;
	this.height = height;
	this.x_speed = 0;
	this.y_speed = 0;
}

Paddle.prototype.render = function(){
	context.fillStyle = "#000000";
	context.fillRect(this.x,this.y,this.width,this.height);
};

//creating objects and functions for two players. One for computer and one for User
function Player(){
	this.paddle = new Paddle(275,580,50,10);
}

function Computer(){
	this.paddle = new Paddle(275,10,50,10);
}

//render the paddle
Player.prototype.render = function(){
	this.paddle.render();
};

//update player
Player.prototype.update = function(){

	for(var key in keysDown){
		var value = Number(key);
		if(value == 37){//left arrow
			this.paddle.move(-4,0);
		}else if(value == 39){//right arrow
			this.paddle.move(4,0);
		}else{
			this.paddle.move(0,0);
		}
	}
};

Paddle.prototype.move = function(x,y){
	this.x += x;
	this.y += y;
	this.x_speed = x;
	this.y_speed = y;

	if(this.x < 0){ //all the way to left
		this.x = 0;
		this.x_speed = 0;
	}else if(this.x + this.width > 400){ // all the way right
		this.x = 400 - this.width;
		this.x_speed = 0;
	}
};


Computer.prototype.render = function(){
	this.paddle.render();
};

//update computer paddle 
Computer.prototype.update = function(ball){

	var x_pos = ball.x;
	var diff = -((this.paddle.x + (this.paddle.width / 2)) - x_pos);

	if (diff < 0 && diff < -4){
		diff -= 5;
	}else if(diff > 0 && diff > 4){
		diff = 5;
	}
	this.paddle.move(diff, 0);
	
	if (this.paddle.x < 0){
		this.paddle.x = 0;
	}else if(this.paddle.x + this.paddle.width > 400){
		this.paddle.x = 400 - this.paddle.width;
	}
};

//custom code. Depending upon what move passed here, move the computer by 4x
//network output is either -1,0, or 1 (left,stay,right)

Computer.prototype.ai_update = function(move =0){
	this.paddle.move(4 * move, 0);
}

//define ball
function Ball(x,y){
	this.x = x;
	this.y = y;
	this.x_speed = 0;
	this.y_speed = 3;
	this.radius = 5;
}

Ball.prototype.render = function(){
	context.beginPath();
	context.arc(this.x,this.y,this.radius,2*Math.PI,false);
	context.fillStyle = "#FF0000";
	context.fill();
};

Ball.prototype.update = function(paddle1,paddle2){
	this.x += this.x_speed;
	this.y += this.y_speed;
	var top_x = this.x - 5;
	var top_y = this.y - 5;
	var bottom_x = this.x + 5;
	var bottom_y = this.y + 5;

	if(this.x - 5 < 0){
		//hitting the left wall
		this.x = 5;
		this.x_speed =  -this.x_speed;
	}else if(this.x+5 > 400){
		//hitting the right wall
		this.x = 395;
		this.x_speed =  -this.x_speed
	}

	if (this.y < 0 || this.y > 600){ //a point was scored
		this.x_speed = 0;
		this.y_speed = 3;
		this.x = 200;
		this.y = 300;
		//calling the function
		ai.new_turn();
	}

	if (top_y > 300){

		if(top_y < (paddle1.y + paddle1.height) && bottom_y > paddle1.y && top_x < (paddle1.x + paddle1.width) && bottom_x > paddle1.x){
			//hit the players padddle
			this.y_speed = -3;
			this.x_speed += (paddle1.x_speed / 2);
			this.y += this.y_speed;
		}
	}else{
		if(top_y < (paddle2.y + paddle2.height) && bottom_y > paddle2.y && top_x < (paddle2.x + paddle2.width) && bottom_x > paddle2.x){
			//hit the computers paddle
			this.y_speed = 3;
			this.x_speed += (paddle2.x_speed / 2);
			this.y += this.y_speed;
		}
	}
};

//custome code
function AI(){
	this.previous_data = null;
	this.training_data = [[],[],[]];
	this.last_data_object = null;
	this.turn = 0;
	this.grab_data = true;
	this.flip_table = true;
}

//saving data per frame
AI.prototype.save_data = function(player,computer,ball){

	if(!this.grab_data){
		return ;
	}

	//if this is the very first frame
	if(this.previous_data == null){
		data = this.flip_table ? [width - computer.x, width - player.x, width - ball.x, height - ball.y] : [player.x, computer.x, ball.x, ball.y];
		this.previous_data = data;
		return;
	}

	//table is rotated to learn from player but apply to computer position
	if(this.flip_table){
		data_xs = [width - computer.x, width - player.x, width - ball.x, height - ball.y];
		index = ((width - player.x) > this.previous_data[1]) ? 0:(((width - player.x) == this.previous_data[1])?1:2);

	}else{
		data_xs = [player.x, computer.x, ball.x, ball.y];
		index = (player.x <  this.previous_data[0])?0:((player.x == this.previous_data[0])?1:2);
	}

	this.last_data_object = [...this.previous_data, ...data_xs];
	this.training_data[index].push(this.last_data_object);
	this.previous_data = data_xs;
}

//AI turn
AI.prototype.new_turn =  function(){
	this.previous_data = null;
	this.turn++;
	console.log('New Turn:' + this.turn);

	if(this.turn >  1){
		this.train();
		computer.ai_plays = true;
		this.reset();
	}
}

//reset the ai play
AI.prototype.reset = function(){
	this.previous_data = null;
	this.training_data = [[],[],[]];
	this.turn = 0;
}

//trains a model
AI.prototype.train =  function(){
	console.log("Balancing...!");
	//shuffle attempt
	len = Math.min(this.training_data[0].length, this.training_data[1].length,this.training_data[2].length);

	if (!len){
		console.log("Nothing to Train");
		return ;
	}

	data_xs = [];
	data_ys = [];

	for(i=0;i<3;i++){
		data_xs.push(...this.training_data[i].slice(0,len));
		data_ys.push(...Array(len).fill([i==0?1:0,i==1?1:0,i==2?1:0]));
	}

	console.log("Training");
	const xs = tf.tensor(data_xs);
	const ys = tf.tensor(data_ys);

	(async function(){
		console.log("Training 2 Phase...");
		let result = await model.fit(xs,ys);
		console.log(result);
	}());

	console.log("Trained");
}

AI.prototype.predict_move = function(){
	console.log("Predicting...!");

	if(this.last_data_object != null){
		//this.last_data_object for input data
		//prediction is done here
		//return -1,0,1

		prediction = model.predict(tf.tensor([this.last_data_object]));

		return tf.argMax(prediction,1).dataSync()-1;
	}
}


//create object for all 
var player = new Player();
var computer = new Computer();
var ball = new Ball(295,300)
var ai = new AI();

//rendering all canvas elements
var render = function(){
	context.fillStyle = "#D3D3D3";
	context.fillRect(0,0,width,height);
	player.render();
	computer.render();
	ball.render();
};

//add key controls for user
var keysDown = {};

//get keycode for particular arrow keys
window.addEventListener("keydown",function(event) {
	keysDown[event.keyCode] = true;
});

//get keycode for upbutton
window.addEventListener("keyup",function(event){
	delete keysDown[event.keyCode];
});


//step function responsible for doing three things
var step = function(){
	update();
	render();
	animate(step);
}

//append to the html body
document.body.appendChild(canvas);
animate(step);