from flask import render_template, url_for, flash, session
from flask import request, redirect
from ownzap import app
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ownzap import model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app.secret_key = b'D\x11\x80\xd2@S\x9b\x86;Sp\xda\xde\xbdB\xc4\xcc\x15&}\x8a\x97\xc2\x17'
information = {
            'GANTA GHAR':['''Ghanta Ghar, also known as the clock tower of Rajasthan, is in the Indian city of Jodhpur.Ghanta ghar is basically the center of the city or you can say the main market area where you can find the tower with an clock which can been visible from quite long distance. You can find a lot of shops around this place for shopping, snakcs & other stuff. ''','1880-1911','Maharaja Sardar Singh'],
            'MEHRANGARH FORT':['''Mehrangarh, located in Jodhpur, Rajasthan, is one of the largest forts in India. The fort is situated 410 feet (125 m) above the city and is enclosed by imposing thick walls. Inside its boundaries there are several palaces known for their intricate carvings and expansive courtyards. A winding road leads to and from the city below. The imprints of the impact of cannonballs fired by attacking armies of Jaipur can still be seen on the second gate. To the left of the fort is the chhatri of Kirat Singh Soda, a soldier who fell on the spot defending Mehrangarh.''','1459','Rao Jodha'],
            'JASWANT TADA':['''The Jaswant Tada is a cenotaph north of Mehranger Fort.  Constructed entirely of thin, polished marble, the sun sits and reflects on the cenotaph’s walls. The grounds include gazebos, a garden, and a small lake.There are three other cenotaphs in the grounds. The cenotaph of Maharaja Jaswant Singh displays portraits of the rulers and Maharajas of Jodhpur.''','1899','Maharaja Sardar Singh'],
            'MANDORE GARDENS':['''Mandore Garden, is a town located 9 km north of Jodhpur city, in the Indian state of Rajasthan.The place is known as the birthplace of Ravana's wife mandodari.Scenic, landscaped grounds featuring tomblike monuments, a temple, statues & the ruins of Mandore.''','6TH CENTURY','Rao Jodha'],
            'UMAID BHAWAN PALACE':['''Umaid Bhawan Palace, located in Jodhpur in Rajasthan, India, is one of the world's largest private residences. A part of the palace is managed by Taj Hotels. Named after Maharaja Umaid Singh, grandfather of the present owner Gaj Singh. The palace has 347 rooms and is the principal residence of the former Jodhpur royal family. A part of the palace is a museum. Ground for the foundations of the building was broken on 18 November 1929. Recently, Umaid Bhawan Palace was awarded as the World's best hotel at the Traveller's Choice Award''','1943','Maharaja Umaid Singh'],
            'GHANTA GHAR':['''Ghanta Ghar, also known as the clock tower of Rajasthan, is in the Indian city of Jodhpur.Ghanta ghar is basically the center of the city or you can say the main market area where you can find the tower with an clock which can been visible from quite long distance. You can find a lot of shops around this place for shopping, snakcs & other stuff. ''','1880-1911','Maharaja Sardar Singh'],
            'TOORJI KA JHALRA':['''Toorji Ka Jhalra (Toorji’s Step Well) was built in Jodhpur the 1740s by a Queen, Maharaja Abhay Singh’s Consort, continuing an age old tradition that Royal women would build public water works.Interestingly this well was submerged and full of debris for decades. Only recently has it been drained, cleaned up and restored. In the process, the excavations went down over two hundred feet to expose hand carved treasures in Jodhpur’s famous rose-red sandstone; including intricate carvings of dancing elephants, medieval lions and cow water-spouts, as well as niches housing deities long gone.''','1740','Maharaja Abhay Singh’s'],
            'KAYLANA LAKE':['''Kaylana Lake is located 8 km west of Jodhpur in Rajasthan, India. It is an artificial lake.The lake spreads over an area of 84 km2. In ancient times this region had palaces and gardens made by two rulers of Jodhpur - Bhim Singh and Takhat Singh. These were destroyed to make Kaylana Lake.
The lake is situated between igneous rock land formations. It receives its water from Hati Nehar (translation: elephant canal), which is further connected to the Indra Gandhi canal. The natural vegetation here mostly consists of Babool trees (Acacia nilotica), and various migratory birds such as Siberian cranes are seen here in the winter season. The city of Jodhpur and all the surrounding towns and villages depend on Kaylana lake as a source of drinking water.''','1872','PRATAP SINGH'],
            'KHEJARLA FORT':['''This fort is in the village of khejarla. This stunning red sandstone monument is a splendid example of Rajput architecture. The contrast of an inner paradise with a spectacular blend of rugged exterior art and architecture is leaving you with a magnificent aura! The enchanted grandeur of the fort offers picturesque settings, latticework friezes, and intricate ventilations that allow you to go back at once to experience the long-time glory of heroism and poise.
The historical structure combines with the golden color of the desert surroundings and the picturesque atmosphere in its most vibrant colors gives it a fairytale look. Nestled in a serene and tranquil environment, the Khejerla Fort holds a deep history and offers guests all the charisma and grandeur of a typical royal experience''','1611','Maharaja Gopal Das Ji'],
            'SARDAR MARKET':['''Sardar Market, one of the oldest Street markets in the middle of the city, is located near the Clock Tower. Built by Maharaja Sardar Singh, this market caters to the countless selection of shopping needs that covers every possible requirement of a person. You can buy almost everything- from handicrafts, clothes, accessories, antiques to spices, fruits and vegetables and much more. Quite a colorful and chaotic market, this place is one of the typical examples of a lively local bazaar. The shopkeepers are simple people, who have been in this occupation for generations. The shopkeepers are always ready for a friendly chat, and if you are lucky enough, you might get a better deal. People get attracted to this market area due to the availability of a variety of items at a much reasonable rate.''','1900','MAHARAJA SARDAR SINGH'],
            'MAHAMANDIR TEMPLE':['''The architectural marvel of Jodhpur, this temple is supported by 84 pillars which are decorated with frescos and carvings depicting yogic postures, intricate motifs and other artwork within its premises. The temple boasts a beautifully designed hall that is used for Yoga classes. The best part of this temple is its royal architecture''','1812','UNKNOWN'],
            }
@app.route("/")
@app.route("/home")
def home():
    try :
        image = os.listdir(app.config['UPLOAD_FOLDER'])[0]
        img  = cv2.imread(app.config['UPLOAD_FOLDER']+'/'+image,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(app.config['UPLOAD_FOLDER']+'/'+'z'+image,img)
        image_for_process = cv2.resize(img,(150,150))
        image_for_model = np.array(image_for_process).reshape((-1,150,150,1))
        print(image_for_model)
        output = predict(image_for_model)
        print(output)
        # print('static/profile_pics'+'/'+image)
        # print(information[output])
        # print('static/profile_pics'+'/'+'z'+image)
        return render_template("home.html",image='static/profile_pics'+'/'+image,image1='static/profile_pics'+'/'+'z'+image,output=output,information = information[output]) 
    except :
        return render_template('home.html')
    # return render_template("home.html")

def predict(image_for_model):
    
    output = model.predict(image_for_model)
    category = ['GANTA GHAR',
    'JASWANT TADA',
    'KAYLANA LAKE',
    'KHEJARLA FORT',
    'MAHAMANDIR TEMPLE',
    'MANDORE GARDENS',
    'MEHRANGARH FORT',
    'SARDAR MARKET',
    'TOORJI KA JHALRA',
    'UMAID BHAWAN PALACE']
    out = map(lambda x:x.upper(),category)
    category = list(out)
    return category[np.argmax(output)]


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
   
    
    if request.method == 'POST':
        try :
            for i in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(app.config['UPLOAD_FOLDER']+'/'+i)
        except :
            pass
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                flash("File didn't uploaded")
                return redirect(url_for('home'))
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File uploaded successfully {}'.format(filename))
            return redirect(url_for('home'))

        flash("File didn't uploaded")
        return redirect(url_for('home'))

    