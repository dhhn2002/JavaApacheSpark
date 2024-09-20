package org.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;

import java.io.Serializable;
import java.util.Collections;

import static org.apache.spark.sql.functions.col;

public class Main {
    public static class Person implements Serializable {
        private String id;
        private String name;
        private long age;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public long getAge() {
            return age;
        }

        public void setAge(long age) {
            this.age = age;
        }
    }

    public static void main(String[] args) throws AnalysisException {
        SparkSession spark = SparkSession
                                .builder()
                                .appName("Java Spark SQL basic example")
                                .master("local[*]")
                                .getOrCreate();

        runBasicDataFrameExample(spark);
        runDatasetCreationExample(spark);
        runInferSchemaExample(spark);

        spark.stop();
    }

    private static void runBasicDataFrameExample(SparkSession spark) throws AnalysisException {
        Dataset<Row> filteredDf = spark.read().json("src/main/resources/people.json");
        Dataset<Row> df = filteredDf.filter(filteredDf.col("name").isNotNull());

        // Displays the content of the DataFrame to stdout
        df.show();

        // Print the schema in a tree format
        df.printSchema();

        // Select only the "name" column
        df.select("name").show();

        // Select everybody, but increment the age by 1
        df.select(col("name"), col("age").plus(1)).show();

        // Select people older than 21
        df.filter(col("age").gt(21)).show();

        // Count people by age
        df.groupBy("age").count().show();

        // Register the DataFrame as a SQL temporary view
        df.createOrReplaceTempView("people");

        Dataset<Row> sqlDF = spark.sql("SELECT * FROM people WHERE age BETWEEN 30 AND 40");
        sqlDF.show();

        // Register the DataFrame as a global temporary view
        df.createGlobalTempView("people");

        // Global temporary view is tied to a system preserved database `global_temp`
        spark.sql("SELECT * FROM global_temp.people").show();

        // Global temporary view is cross-session
        spark.newSession().sql("SELECT * FROM global_temp.people").show();
    }

    private static void runDatasetCreationExample(SparkSession spark) {
        // Create an instance of a Bean class
        Person person = new Person();
        person.setName("Andy");
        person.setAge(32);

        // Encoders are created for Java beans
        Encoder<Person> personEncoder = Encoders.bean(Person.class);
        Dataset<Person> javaBeanDS = spark.createDataset(
                Collections.singletonList(person),
                personEncoder
        );
        javaBeanDS.show();

//        // Encoders for most common types are provided in class Encoders
//        Encoder<Long> longEncoder = Encoders.LONG();
//        Dataset<Long> primitiveDS = spark.createDataset(Arrays.asList(1L, 2L, 3L), longEncoder);
//        Dataset<Long> transformedDS = primitiveDS.map(
//                (MapFunction<Long, Long>) value -> value + 1L,
//                longEncoder);
//        transformedDS.collect(); // Returns [2, 3, 4]

        // DataFrames can be converted to a Dataset by providing a class. Mapping based on name
        String path = "src/main/resources/people.json";
        Dataset<Person> peopleDS = spark.read().json(path).as(personEncoder);
        peopleDS.show();
    }

    private static void runInferSchemaExample(SparkSession spark) {
        // Create an RDD of Person objects from a text file
        JavaRDD<Person> peopleRDD = spark.read()
                .textFile("src/main/resources/people.txt")
                .javaRDD()
                .map(line -> {
                    String[] parts = line.split(",");
                    Person person = new Person();
                    person.setName(parts[0]);
                    person.setAge(Integer.parseInt(parts[1].trim()));
                    return person;
                });

        // Apply a schema to an RDD of JavaBeans to get a DataFrame
        Dataset<Row> peopleDF = spark.createDataFrame(peopleRDD, Person.class);
        // Register the DataFrame as a temporary view
        peopleDF.createOrReplaceTempView("people");

        // SQL statements can be run by using the sql methods provided by spark
        Dataset<Row> teenagersDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 15 AND 40");

        // The columns of a row in the result can be accessed by field index
        Encoder<String> stringEncoder = Encoders.STRING();
        Dataset<String> teenagerNamesByIndexDF = teenagersDF.map(
                (MapFunction<Row, String>) row -> "Name: " + row.getString(0),
                stringEncoder);
        teenagerNamesByIndexDF.show();
        // +------------+
        // |       value|
        // +------------+
        // |Name: Justin|
        // +------------+

        // or by field name
        Dataset<String> teenagerNamesByFieldDF = teenagersDF.map(
                (MapFunction<Row, String>) row -> "Name: " + row.<String>getAs("name"),
                stringEncoder);
        teenagerNamesByFieldDF.show();
        // +------------+
        // |       value|
        // +------------+
        // |Name: Justin|
        // +------------+
    }

}