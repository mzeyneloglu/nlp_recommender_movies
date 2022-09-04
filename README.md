# NLP Recommender Movies

Bir film izlediğimizde izlediğimiz filme benzer olanları tavsiye etmesini istiyoruz bunun için  **Vector Similarity** yapısını kullanarak gerçekleştiriyor olacağız. Peki Vector Similarity nedir ?

### Vector Similarity


<img src="https://miro.medium.com/max/1200/1*vjxvrsTNhJ92IF-N95uIOQ.png"  width="600" height="400">

Dili makine tarafından okunabilir bir biçime dönüştürdüğümüzde yani vektörlere dönüştürdüğümüzde,  standart yaklaşım yoğun vektörler kullanmaktır.Bir sinir ağı tipik olarak yoğun vektörler üretir. Sözcükleri ve cümleleri, her vektörün geometrik konumunun anlam yükleyebileceği şekilde düzenlenmiş, yüksek boyutlu vektörlere dönüştürmemize izin veriyorlar.

<img src="https://miro.medium.com/max/700/0*CBxUXv7niY5hS9TH.png"  width="600" height="400">

Bunun özellikle iyi bilinen bir örneği vardır, burada **King** vektörünü alırız, **Man** vektörünü çıkarırız ve **Kadın** vektörünü ekleriz. Elde edilen vektöre en yakın eşleşen vektör **Kraliçe**'dir.Aynı mantığı cümleler veya paragraflar gibi daha uzun dizilere de uygulayabiliriz - ve benzer anlamın bu vektörler arasındaki yakınlık/yönelime karşılık geldiğini bulacağız.Dolayısıyla benzerlik önemlidir - ve burada ele alacağımız şey, bu benzerliği hesaplamak için en popüler iki ölçüttür.

- Euclidean Distance
- Cosine Similarity

#### Euclidean Distance ( Oklid Mesafesi )


Öklid mesafesi (genellikle L2 normu olarak adlandırılır) metriklerin en sezgisel olanıdır. Üç vektör tanımlayalım:

<img src="https://miro.medium.com/proxy/1*zkJlmLrKyzSgae6PUypwxQ.png"  width="600" height="300">
 
Sadece bu vektörlere bakarak, **a** ve **b**'nin birbirine daha yakın olduğunu güvenle söyleyebiliriz – ve her birini bir grafikte görselleştirirken bunu daha da net görürür.

<img src="https://miro.medium.com/proxy/1*0PuxQcmsuL2pRVFJBt2cHA.png"  width="600" height="300">

Vektörler **a** ve **b** orijinine yakındır, vektör **c** çok daha uzaktır.Açıkçası, **a** ve **b** birbirine daha yakındır - ve bunu Öklid mesafesini kullanarak hesaplıyoruz:

<img src="https://miro.medium.com/proxy/1*X5MMsxJauDXDh3RKnJKWLQ.png"  width="800" height="200">
 
Öklid mesafe formülü,bu formülü iki vektörümüze, **a** ve **b,** uygulamak için şunları yaparız:

<img src="https://miro.medium.com/proxy/1*nvG_GpVyjMUcMz5R7M7xHA.png"  width="800" height="200">

**a** ve **b** vektörleri arasındaki Öklid mesafesinin hesaplanması ve **0,014** mesafe elde ediyoruz, **d(a, c)** için aynı hesaplamayı yapıyoruz **1.145** ve **d(b, c)** **1.136** döndürüyor. Açıkçası, **a** ve **b** Öklid uzayında daha yakındır.

> Oklid mesefasi ne kadar düşük ise o kadar benzerlik yüksek anlamına gelmektedir.

#### Cosine Similarity ( Kosinüs Benzerliği )
Kosinüs benzerliği, vektör büyüklüğünden bağımsız olarak vektör oryantasyonunu dikkate alır.
<img src="https://miro.medium.com/max/700/1*BJq5ZVsO4rpYTsmV9pIGUQ.png" width="800" height="200">


Kosinüs benzerlik formülü
Bu formülde dikkat etmemiz gereken ilk şey, payın aslında hem büyüklüğü hem de yönü dikkate alan nokta ürünü olduğudur.

Paydada, garip çift dikey çubuklarımız var - bunlar 'uzunluğu' anlamına geliyor. Böylece, sizin uzunluğunuz v'nin uzunluğuyla çarpılır. Uzunluk, elbette, büyüklüğü dikkate alır.

Hem büyüklüğü hem de yönü dikkate alan bir fonksiyonu alıp bunu sadece büyüklüğü dikkate alan bir fonksiyona böldüğümüzde – bu iki büyüklük ortadan kalkar ve bizi yönü büyüklükten bağımsız olarak gören bir fonksiyonla bırakır.
<img src="https://miro.medium.com/max/700/1*SWK3921MItHvJEoefHhJsw.png"  width="800" height="200">


Kosinüs benzerliğini normalleştirilmiş bir nokta ürünü olarak düşünebiliriz! Ve açıkça işe yarıyor. A ve b'nin kosinüs benzerliği 1'e yakındır (mükemmel):

A ve b vektörleri için kosinüs benzerliğinin hesaplanması ve a ve c'yi tekrar karşılaştırmak için kosinüs benzerliğinin **sklearn** uygulamasını kullanmak bize çok daha iyi sonuçlar verir:


<img src="https://miro.medium.com/max/700/1*5vhMrw6lgUGQj9HtYb5W1w.png"  width="800" height="200">

Vector similarity uygulamaları NLP içerisinde oldukça sık kullanılmaktadır.Bir sözcüğün bir sözcüğe yakınlığını hesaplamak gibi veya bir metin yazım işlemi gerçekleştirilirken yazılan metinlerden farklı bir sonuç elde etmek gibi bir çok örnek verebiliriz.


Vector Similarity ne olduğuna baktığımıza göre TfidfVectorizer teorik olarak göz gezdirdikten sonra işe koyulalım :)


### TfidfVectorizer 
 
Makine öğrenimi algoritmaları genellikle sayısal verileri kullanır, bu nedenle metinsel verilerle veya metinle ilgilenen ML / AI'nın bir alt alanı olan herhangi bir doğal dil işleme (NLP) göreviyle uğraşırken, bu verilerin önce vektörleştirme olarak bilinen bir işlemle sayısal veri vektörüne dönüştürülmesi gerekir. **TF-IDF vektörizasyonu, derleminizdeki her kelime için TF-IDF puanını bu belgeye göre hesaplamayı ve ardından bu bilgiyi bir vektöre koymayı içerir ('A' ve 'B' örnek belgelerini kullanarak aşağıdaki resme bakın). Böylece derleminizdeki her belgenin kendi vektörü olacaktır ve vektörün tüm belge koleksiyonundaki her bir kelime için bir TF-IDF puanı olacaktır. Bu vektörlere sahip olduktan sonra, bunları kosinüs benzerliğini kullanarak TF-IDF vektörlerini karşılaştırarak iki belgenin benzer olup olmadığını(vector similarity) görmek gibi çeşitli kullanım durumlarına uygulayabilirsiniz.**


<img src="https://ecm.capitalone.com/WCM/tech/tf-idf-4.png"  width="600" height="400">

> Kullanım şekli CounVectorizer gibi sklearn içerisinden aktarıldıktan sonra veri seti üzerinde fit ve transform edilmesi ile gerçekleşir.TfidfVectorizer , CounVectorizer işlevinin gelmiş halidir.

TF-IDF'nin ayrıca bilgi alma alanında kullanım durumları vardır, yaygın bir örnek arama motorlarıdır. TF-IDF, bir belgeye dayalı bir terimin ilgili önemi hakkında size bilgi verebildiğinden, bir arama motoru, arama sonuçlarını alaka düzeyine göre sıralamaya yardımcı olmak için TF-IDF'yi kullanabilir ve sonuçlar daha yüksek TF-IDF puanlarına sahip kullanıcıyla daha alakalı sonuçlar elde edebilir.

Dikkat edilmesi gereken bir şey, TF-IDF'nin semantik anlam taşımaya yardımcı olamayacağıdır. Kelimelerin önemini, onları nasıl tarttığına bağlı olarak düşünür, ancak kelimelerin bağlamlarını mutlaka türetemez ve önemi bu şekilde anlayamaz.

Ayrıca yukarıda belirtildiği gibi, BoW gibi, tf-idf kelime sırasını görmezden gelir ve bu nedenle 'İngiltere Kraliçesi' gibi bileşik isimler 'tek bir birim' olarak kabul edilmez. Bu aynı zamanda, siparişin büyük bir fark yarattığı 'faturayı ödememek' ve 'faturayı ödemek' ile reddetme gibi durumlara da uzanır. Her iki durumda da NER araçlarını ve alt çizgilerini kullanarak, 'queen_of_england' veya 'not_pay', ifadeyi tek bir birim olarak ele almanın yollarıdır.

Diğer bir dezavantaj, TF-IDF boyutsallığın lanetinden muzdarip olabileceğinden, bellek verimsizliğinden muzdarip olabilmesidir. TF-IDF vektörlerinin uzunluğunun kelime dağarcığının boyutuna eşit olduğunu hatırlayın. Bazı sınıflandırma bağlamlarında bu bir sorun olmayabilir, ancak kümeleme gibi diğer bağlamlarda belge sayısı arttıkça bu hantal olabilir. Bu nedenle,n alternatiflerden bazılarına **(BERT, Word2Vec)** bakmak gerekli olabilir.
