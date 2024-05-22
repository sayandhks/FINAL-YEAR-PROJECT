#define measurePin  0//Connect dust sensor to Arduino A0 pin
#define ledPower  7   //Connect 3 led driver pins of dust sensor to Arduino D2
int samplingTime = 280; // time required to sample signal coming out   of the sensor
int deltaTime = 40; // 
int sleepTime = 9680;
float voMeasured = 0;
float calcVoltage = 0;
float dustDensity = 0;
int sensorPin=A1;
int sensorData;
int sensorPin1=A2;
int sensorData1;
void setup(){

  Serial.begin(9600);
  pinMode(ledPower,OUTPUT);
  pinMode(sensorPin,INPUT);   
  pinMode(sensorPin1,INPUT);

}


void loop(){
  sensorData = analogRead(sensorPin);       
  sensorData1 = analogRead(sensorPin1);   
  digitalWrite(ledPower,LOW); // power on the LED
  delayMicroseconds(samplingTime);
  voMeasured = analogRead(measurePin); // read the dust value
  delayMicroseconds(deltaTime);
  digitalWrite(ledPower,HIGH); // turn the LED off
  delayMicroseconds(sleepTime);
  calcVoltage = voMeasured * (5.0 / 1024.0);
  dustDensity = 170 * calcVoltage - 0.1;
  Serial.print(dustDensity); // unit: ug/m3
  delay(1000);
  Serial.print(",");
  Serial.print(sensorData);
  Serial.print(",");    
  Serial.println(sensorData1);

}