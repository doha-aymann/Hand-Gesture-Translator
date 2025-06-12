//Arduino code

#include<LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,16,2);
void setup() {
 
  lcd.init();
  lcd.backlight();
  Serial.begin(115200);

}

void loop() {

if(Serial.available()){
Serial.println("done");
String letter = Serial.readStringUntil('\n');
letter.trim();
if(letter =="#"){
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("last letter ");
  lcd.setCursor(0,1);
 lcd.print("is deleted ");
 delay(1000);
}else if(letter == "*"){
   lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Reset");
  delay(1000);
}else{

  lcd.clear();
  lcd.setCursor(0,1);
  lcd.print(letter);
  delay(1000);
}


}





}