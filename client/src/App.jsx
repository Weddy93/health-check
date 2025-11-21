import React, { useState, useEffect } from 'react';
 
// Health Assessment Logic
class HealthAssessor {
  calculateBMI(weight, height) {
    const heightInMeters = height / 100;
    return weight / (heightInMeters * heightInMeters);
  }

  calculateBMR(weight, height, age, gender) {
    if (gender === 'male') {
      return 10 * weight + 6.25 * height - 5 * age + 5;
    } else {
      return 10 * weight + 6.25 * height - 5 * age - 161;
    }
  }

  assessHealth(profile) {
    const bmi = this.calculateBMI(profile.weight, profile.height);
    const bmr = this.calculateBMR(profile.weight, profile.height, profile.age, profile.gender);
    const symptoms = profile.symptoms;

    let healthStatus = 'healthy';
    let diagnosis = '';
    let recommendations = [];
    let nutrientsNeeded = [];

    // BMI Assessment
    let bmiCategory = '';
    if (bmi < 18.5) {
      bmiCategory = 'underweight';
      healthStatus = 'malnourished';
      nutrientsNeeded.push('calories', 'protein', 'healthy fats');
    } else if (bmi >= 18.5 && bmi <= 24.9) {
      bmiCategory = 'normal';
    } else if (bmi >= 25 && bmi <= 29.9) {
      bmiCategory = 'overweight';
      healthStatus = 'malnourished';
      nutrientsNeeded.push('fiber', 'lean protein');
    } else {
      bmiCategory = 'obese';
      healthStatus = 'malnourished';
      nutrientsNeeded.push('fiber', 'lean protein', 'complex carbs');
    }

    // Symptom-based assessment
    if (symptoms.includes('fatigue') && symptoms.includes('weakness')) {
      diagnosis += 'Possible Iron Deficiency Anemia. ';
      nutrientsNeeded.push('iron', 'vitamin B12', 'vitamin C');
    }

    if (symptoms.includes('hairloss') && symptoms.includes('skin')) {
      diagnosis += 'Potential Zinc or Biotin Deficiency. ';
      nutrientsNeeded.push('zinc', 'biotin', 'vitamin A');
    }

    if (symptoms.includes('bones') && symptoms.length >= 2) {
      diagnosis += 'Possible Vitamin D or Calcium Deficiency. ';
      nutrientsNeeded.push('vitamin D', 'calcium', 'magnesium');
    }

    if (symptoms.includes('weight-loss') && symptoms.length >= 3) {
      diagnosis += 'Potential Protein-Energy Malnutrition. ';
      nutrientsNeeded.push('protein', 'calories', 'essential fatty acids');
    }

    // Diet-based assessment
    if (profile.diet === 'unhealthy') {
      diagnosis += 'Poor dietary choices detected. ';
      nutrientsNeeded.push('fiber', 'vitamins', 'minerals', 'antioxidants');
    }

    if (profile.diet === 'vegan' && symptoms.includes('weakness')) {
      diagnosis += 'Monitor Vitamin B12 and Iron levels. ';
      nutrientsNeeded.push('vitamin B12', 'iron', 'omega-3');
    }

    // Remove duplicates from nutrients needed
    nutrientsNeeded = [...new Set(nutrientsNeeded)];

    // Generate food recommendations based on nutrients needed
    const foodRecommendations = this.generateFoodRecommendations(nutrientsNeeded);

    // If no specific diagnosis but still malnourished based on BMI
    if (diagnosis === '' && healthStatus === 'malnourished') {
      diagnosis = `Weight-related nutritional imbalance (${bmiCategory}).`;
    }

    // If healthy but has some symptoms
    if (healthStatus === 'healthy' && symptoms.length > 0) {
      diagnosis = 'Generally healthy but showing minor symptoms. Consider consulting a nutritionist.';
    }

    return {
      healthStatus,
      bmi: bmi.toFixed(1),
      bmiCategory,
      diagnosis: diagnosis || 'No specific nutritional deficiencies detected.',
      recommendations: foodRecommendations,
      nutrientsNeeded
    };
  }

  generateFoodRecommendations(nutrients) {
    const foodMap = {
      'iron': ['Spinach', 'Red meat', 'Lentils', 'Fortified cereals', 'Beans'],
      'protein': ['Chicken breast', 'Fish', 'Eggs', 'Greek yogurt', 'Tofu', 'Lentils'],
      'calories': ['Nuts', 'Avocado', 'Whole grains', 'Healthy oils', 'Dried fruits'],
      'vitamin B12': ['Animal liver', 'Clams', 'Fortified cereals', 'Nutritional yeast'],
      'vitamin C': ['Citrus fruits', 'Bell peppers', 'Broccoli', 'Strawberries'],
      'zinc': ['Oysters', 'Beef', 'Pumpkin seeds', 'Chickpeas'],
      'biotin': ['Eggs', 'Almonds', 'Sweet potatoes', 'Spinach'],
      'vitamin A': ['Carrots', 'Sweet potatoes', 'Kale', 'Eggs'],
      'vitamin D': ['Fatty fish', 'Fortified milk', 'Egg yolks', 'Sunlight exposure'],
      'calcium': ['Dairy products', 'Fortified plant milk', 'Kale', 'Broccoli'],
      'magnesium': ['Nuts', 'Seeds', 'Whole grains', 'Dark chocolate'],
      'fiber': ['Whole grains', 'Vegetables', 'Fruits', 'Legumes', 'Nuts'],
      'healthy fats': ['Avocado', 'Nuts', 'Olive oil', 'Fatty fish'],
      'complex carbs': ['Oats', 'Quinoa', 'Brown rice', 'Sweet potatoes'],
      'omega-3': ['Fatty fish', 'Flaxseeds', 'Walnuts', 'Chia seeds'],
      'antioxidants': ['Berries', 'Dark leafy greens', 'Nuts', 'Dark chocolate']
    };

    let recommendations = [];
    nutrients.forEach(nutrient => {
      if (foodMap[nutrient]) {
        recommendations.push({
          nutrient,
          foods: foodMap[nutrient]
        });
      }
    });

    return recommendations;
  }
}

// Meal Plan Generator
class MealPlanGenerator {
  constructor() {
    this.mealTemplates = {
      highProtein: {
        breakfast: ['Scrambled eggs with spinach', 'Whole grain toast', 'Greek yogurt'],
        lunch: ['Grilled chicken salad', 'Quinoa', 'Mixed vegetables'],
        dinner: ['Baked salmon', 'Steamed broccoli', 'Sweet potato'],
        snack: ['Protein shake', 'Handful of almonds']
      },
      ironRich: {
        breakfast: ['Fortified cereal with berries', 'Orange juice', 'Hard-boiled eggs'],
        lunch: ['Spinach and lentil soup', 'Whole grain bread', 'Lean red meat'],
        dinner: ['Beef stir-fry with bell peppers', 'Brown rice', 'Steamed kale'],
        snack: ['Dried apricots', 'Pumpkin seeds']
      },
      energyBoost: {
        breakfast: ['Oatmeal with bananas', 'Walnuts', 'Milk'],
        lunch: ['Chicken and avocado wrap', 'Fruit salad', 'Yogurt'],
        dinner: ['Turkey meatballs', 'Whole wheat pasta', 'Green beans'],
        snack: ['Apple with peanut butter', 'Trail mix']
      },
      weightGain: {
        breakfast: ['Peanut butter smoothie', 'Bagel with cream cheese', 'Banana'],
        lunch: ['Chicken and rice bowl', 'Avocado', 'Mixed nuts'],
        dinner: ['Salmon with quinoa', 'Roasted vegetables', 'Olive oil dressing'],
        snack: ['Cottage cheese with fruit', 'Granola bars']
      },
      balanced: {
        breakfast: ['Whole grain cereal', 'Milk', 'Fresh fruit'],
        lunch: ['Vegetable soup', 'Sandwich with lean protein', 'Side salad'],
        dinner: ['Baked fish', 'Steamed vegetables', 'Brown rice'],
        snack: ['Yogurt', 'Fresh fruit']
      }
    };

    this.foodCategories = {
      proteins: ['Chicken breast', 'Salmon', 'Eggs', 'Greek yogurt', 'Lentils', 'Tofu'],
      vegetables: ['Spinach', 'Broccoli', 'Bell peppers', 'Carrots', 'Kale', 'Tomatoes'],
      fruits: ['Berries', 'Bananas', 'Oranges', 'Apples', 'Avocado'],
      grains: ['Quinoa', 'Brown rice', 'Oats', 'Whole grain bread', 'Whole wheat pasta'],
      dairy: ['Milk', 'Yogurt', 'Cheese', 'Cottage cheese'],
      nutsSeeds: ['Almonds', 'Walnuts', 'Chia seeds', 'Flaxseeds', 'Pumpkin seeds'],
      healthyFats: ['Olive oil', 'Avocado oil', 'Coconut oil', 'Nut butters']
    };
  }

  generateMealPlan(nutrientsNeeded, duration = 7) {
    let planType = 'balanced';

    // Determine plan type based on nutrients needed
    if (nutrientsNeeded.includes('protein')) {
      planType = 'highProtein';
    } else if (nutrientsNeeded.includes('iron')) {
      planType = 'highProtein'; // Iron-rich foods often overlap with protein
    } else if (nutrientsNeeded.includes('calories')) {
      planType = 'weightGain';
    }

    const weeklyPlan = [];
    const baseMeals = this.mealTemplates[planType];

    for (let day = 1; day <= duration; day++) {
      const dailyMeals = {
        day: `Day ${day}`,
        meals: {
          breakfast: this.varyMeal(baseMeals.breakfast, day),
          lunch: this.varyMeal(baseMeals.lunch, day),
          dinner: this.varyMeal(baseMeals.dinner, day),
          snack: this.varyMeal(baseMeals.snack, day)
        }
      };
      weeklyPlan.push(dailyMeals);
    }

    return weeklyPlan;
  }

  varyMeal(baseMeals, day) {
    // Simple variation logic
    const variations = [
      ['Add different herbs and spices'],
      ['Try different cooking methods'],
      ['Include seasonal vegetables'],
      ['Add a side salad'],
      ['Use different protein sources'],
      ['Include different grains'],
      ['Add healthy sauces or dressings']
    ];

    const variedMeals = [...baseMeals];
    if (variations[day - 1]) {
      variedMeals.push(variations[day - 1][0]);
    }

    return variedMeals;
  }

  generateShoppingList(weeklyPlan) {
    const shoppingList = {};

    weeklyPlan.forEach(day => {
      Object.values(day.meals).forEach(mealItems => {
        mealItems.forEach(item => {
          // Categorize items (simplified)
          let category = 'other';
          const itemLower = item.toLowerCase();

          if (this.containsAny(itemLower, ['chicken', 'fish', 'egg', 'meat', 'tofu', 'lentil'])) {
            category = 'proteins';
          } else if (this.containsAny(itemLower, ['spinach', 'broccoli', 'vegetable', 'salad', 'kale', 'pepper'])) {
            category = 'vegetables';
          } else if (this.containsAny(itemLower, ['fruit', 'berry', 'banana', 'apple', 'orange', 'avocado'])) {
            category = 'fruits';
          } else if (this.containsAny(itemLower, ['grain', 'rice', 'quinoa', 'oat', 'bread', 'pasta'])) {
            category = 'grains';
          } else if (this.containsAny(itemLower, ['milk', 'yogurt', 'cheese'])) {
            category = 'dairy';
          } else if (this.containsAny(itemLower, ['nut', 'seed', 'almond'])) {
            category = 'nutsSeeds';
          }

          if (!shoppingList[category]) {
            shoppingList[category] = new Set();
          }
          // Extract main ingredient (simplified)
          const mainIngredient = this.extractMainIngredient(item);
          if (mainIngredient) {
            shoppingList[category].add(mainIngredient);
          }
        });
      });
    });

    // Convert Sets to Arrays
    const result = {};
    Object.keys(shoppingList).forEach(category => {
      result[category] = Array.from(shoppingList[category]);
    });

    return result;
  }

  containsAny(str, terms) {
    return terms.some(term => str.includes(term));
  }

  extractMainIngredient(item) {
    // Simple extraction
    const words = item.toLowerCase().split(' ');
    const ignoreWords = new Set(['with', 'and', 'the', 'a', 'an', 'in', 'on', 'of', 'for']);

    for (let word of words) {
      if (!ignoreWords.has(word) && word.length > 3) {
        // Capitalize first letter
        return word.charAt(0).toUpperCase() + word.slice(1);
      }
    }
    return item;
  }
}

function App() {
  const [currentPage, setCurrentPage] = useState('profile');
  const [assessment, setAssessment] = useState(null);
  const [profile, setProfile] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [mealPlan, setMealPlan] = useState(null);
  const [shoppingList, setShoppingList] = useState(null);

  const healthAssessor = new HealthAssessor();
  const mealPlanGenerator = new MealPlanGenerator();

  const handleFormSubmit = (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const symptoms = Array.from(e.target.querySelectorAll('input[name="symptoms"]:checked'))
      .map(checkbox => checkbox.value);

    // Add other symptoms if provided
    const otherSymptoms = formData.get('otherSymptoms');
    if (otherSymptoms && otherSymptoms.trim()) {
      // Split by commas and add to symptoms array
      const additionalSymptoms = otherSymptoms.split(',')
        .map(s => s.trim().toLowerCase().replace(/\s+/g, ''))
        .filter(s => s.length > 0);
      symptoms.push(...additionalSymptoms);
    }

    const profileData = {
      name: formData.get('name'),
      age: parseInt(formData.get('age')),
      gender: formData.get('gender'),
      weight: parseFloat(formData.get('weight')),
      height: parseInt(formData.get('height')),
      activity: formData.get('activity'),
      diet: formData.get('diet'),
      symptoms: symptoms
    };

    // Use local health assessment
    const assessmentResult = healthAssessor.assessHealth(profileData);
    setAssessment(assessmentResult);
    setProfile(profileData);
    setShowPopup(true);
  };

  const handleViewMealPlan = () => {
    if (assessment) {
      // Use local meal plan generation
      const plan = mealPlanGenerator.generateMealPlan(assessment.nutrientsNeeded);
      const list = mealPlanGenerator.generateShoppingList(plan);
      setMealPlan(plan);
      setShoppingList(list);
      setCurrentPage('mealPlan');
      setShowPopup(false);
    }
  };

  const handleNewAssessment = () => {
    setAssessment(null);
    setProfile(null);
    setMealPlan(null);
    setShoppingList(null);
    setCurrentPage('profile');
    setShowPopup(false);
  };

  const renderProfilePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 p-5">
      <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
        <header className="bg-gradient-to-r from-green-500 to-green-600 text-white p-12 text-center">
          <h1 className="text-5xl font-bold mb-4">🍎 NutriCheck</h1>
          <p className="text-xl">Complete Health Assessment & Personalized Nutrition</p>
        </header>

        <div className="p-12">
          <h2 className="text-3xl text-gray-800 mb-8 text-center">Personal Health Profile</h2>
          <form onSubmit={handleFormSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Full Name:</label>
                <input
                  type="text"
                  name="name"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Age:</label>
                <input
                  type="number"
                  name="age"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Gender:</label>
                <select
                  name="gender"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                >
                  <option value="">Select</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Weight (kg):</label>
                <input
                  type="number"
                  name="weight"
                  step="0.1"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Height (cm):</label>
                <input
                  type="number"
                  name="height"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Activity Level:</label>
                <select
                  name="activity"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                >
                  <option value="">Select</option>
                  <option value="sedentary">Sedentary (little or no exercise)</option>
                  <option value="light">Light (exercise 1-3 times/week)</option>
                  <option value="moderate">Moderate (exercise 3-5 times/week)</option>
                  <option value="active">Active (daily exercise)</option>
                  <option value="very-active">Very Active (intense daily exercise)</option>
                </select>
              </div>
              <div>
                <label className="block mb-2 font-semibold text-gray-700">Current Diet Type:</label>
                <select
                  name="diet"
                  required
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors"
                >
                  <option value="">Select</option>
                  <option value="balanced">Balanced</option>
                  <option value="vegetarian">Vegetarian</option>
                  <option value="vegan">Vegan</option>
                  <option value="keto">Keto</option>
                  <option value="high-protein">High Protein</option>
                  <option value="unhealthy">Unhealthy/Junk Food</option>
                </select>
              </div>
            </div>

            <div className="bg-gray-50 p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Common Symptoms (Select all that apply):</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { label: 'Fatigue/Low Energy', value: 'fatigue' },
                  { label: 'Muscle Weakness', value: 'weakness' },
                  { label: 'Hair Loss', value: 'hairloss' },
                  { label: 'Skin Problems', value: 'skin' },
                  { label: 'Unexplained Weight Loss', value: 'weight-loss' },
                  { label: 'Unexplained Weight Gain', value: 'weight-gain' },
                  { label: 'Bone/Joint Pain', value: 'bones' },
                  { label: 'Digestive Issues', value: 'digestive' }
                ].map(symptom => (
                  <label key={symptom.label} className="flex items-center font-normal cursor-pointer">
                    <input
                      type="checkbox"
                      name="symptoms"
                      value={symptom.value}
                      className="mr-3 w-4 h-4 text-green-600 bg-gray-100 border-gray-300 rounded focus:ring-green-500"
                    />
                    {symptom.label}
                  </label>
                ))}
              </div>

              <div className="mt-6">
                <label className="block mb-2 font-semibold text-gray-700">Other Symptoms (Optional - Describe any additional symptoms):</label>
                <textarea
                  name="otherSymptoms"
                  placeholder="e.g., frequent headaches, dizziness, poor concentration, etc."
                  className="w-full p-3 border-2 border-gray-300 rounded-xl text-lg focus:border-green-500 focus:outline-none transition-colors resize-none"
                  rows="3"
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-gradient-to-r from-green-500 to-green-600 text-white py-4 px-6 rounded-xl text-xl font-semibold hover:from-green-600 hover:to-green-700 transform hover:scale-105 transition-all duration-200"
            >
              Analyze My Health
            </button>
          </form>

          {assessment && (
            <div className="mt-12 bg-gray-50 p-8 rounded-xl">
              <h2 className="text-3xl text-gray-800 mb-6 text-center">Health Assessment Results</h2>
              <div className="text-center p-8 rounded-2xl mb-8" style={{
                background: assessment.healthStatus === 'healthy'
                  ? 'linear-gradient(135deg, #d4edda, #c3e6cb)'
                  : 'linear-gradient(135deg, #f8d7da, #f5c6cb)',
                border: assessment.healthStatus === 'healthy'
                  ? '2px solid #c3e6cb'
                  : '2px solid #f5c6cb'
              }}>
                <div className="text-6xl mb-4">{assessment.healthStatus === 'healthy' ? '😊' : '⚠️'}</div>
                <h3 className="text-3xl font-bold mb-2">{assessment.healthStatus === 'healthy' ? 'Excellent Health!' : 'Attention Needed!'}</h3>
                <p className="text-xl"><strong>BMI:</strong> {assessment.bmi} ({assessment.bmiCategory})</p>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h3 className="text-2xl font-semibold text-gray-800 mb-4">Assessment Details</h3>
                <p className="text-lg mb-4"><strong>Diagnosis:</strong> {assessment.diagnosis}</p>

                {assessment.nutrientsNeeded && assessment.nutrientsNeeded.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-xl font-semibold text-red-600 mb-3">Recommended Nutrients:</h4>
                    <ul className="list-disc list-inside space-y-2">
                      {assessment.nutrientsNeeded.map(nutrient => (
                        <li key={nutrient} className="text-lg">{nutrient}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {assessment.recommendations && assessment.recommendations.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-xl font-semibold text-blue-600 mb-3">Food Recommendations:</h4>
                    <ul className="list-disc list-inside space-y-2">
                      {assessment.recommendations.map((rec, index) => (
                        <li key={index} className="text-lg"><strong>{rec.nutrient}:</strong> {rec.foods.join(', ')}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="flex justify-center gap-4 mt-8">
                {assessment.healthStatus !== 'healthy' && (
                  <button
                    onClick={handleViewMealPlan}
                    className="bg-gradient-to-r from-red-500 to-red-600 text-white py-3 px-8 rounded-2xl text-lg font-semibold hover:from-red-600 hover:to-red-700 transform hover:scale-105 transition-all duration-200"
                  >
                    View Meal Plan
                  </button>
                )}
                <button
                  onClick={handleNewAssessment}
                  className="bg-gradient-to-r from-gray-500 to-gray-600 text-white py-3 px-8 rounded-2xl text-lg font-semibold hover:from-gray-600 hover:to-gray-700 transform hover:scale-105 transition-all duration-200"
                >
                  New Assessment
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderMealPlanPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 p-5">
      <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
        <header className="bg-gradient-to-r from-orange-500 to-red-500 text-white p-12 text-center">
          <h1 className="text-5xl font-bold mb-4">🍽️ Your Personalized Meal Plan</h1>
          <p className="text-xl">Tailored nutrition based on your health assessment</p>
        </header>

        <div className="p-12">
          {mealPlan && assessment && (
            <>
              <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-8 rounded-2xl mb-12 text-center">
                <h2 className="text-3xl font-bold text-gray-800 mb-4">Your Custom Nutrition Plan</h2>
                <p className="text-xl mb-6">Focusing on: <strong>{assessment.nutrientsNeeded.join(', ')}</strong></p>
                {assessment.nutrientsNeeded.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                    {assessment.nutrientsNeeded.map(nutrient => (
                      <div key={nutrient} className="bg-white p-6 rounded-xl shadow-lg">
                        <h4 className="text-xl font-semibold text-red-600 mb-2">{nutrient.charAt(0).toUpperCase() + nutrient.slice(1)}</h4>
                        <p className="text-gray-600">Essential for your health goals</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="mb-12">
                <h2 className="text-4xl font-bold text-center text-gray-800 mb-8">7-Day Nutrition Plan</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {mealPlan.map(day => (
                    <div key={day.day} className="bg-white border-2 border-gray-200 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300">
                      <h3 className="text-2xl font-bold text-red-600 mb-6 text-center">{day.day}</h3>
                      <div className="space-y-6">
                        {Object.entries(day.meals).map(([mealType, items]) => (
                          <div key={mealType} className="border-b border-gray-100 pb-4 last:border-b-0">
                            <h4 className="text-lg font-semibold text-gray-800 mb-3 capitalize flex items-center">
                              <span className="mr-2">
                                {mealType === 'breakfast' ? '🌅' :
                                 mealType === 'lunch' ? '🍽️' :
                                 mealType === 'dinner' ? '🌙' : '🥨'}
                              </span>
                              {mealType}
                            </h4>
                            <ul className="list-disc list-inside space-y-1 text-gray-700">
                              {items.map((item, index) => (
                                <li key={index}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {shoppingList && (
                <div className="bg-gray-50 p-8 rounded-2xl">
                  <h2 className="text-4xl font-bold text-center text-gray-800 mb-8">🛒 Recommended Shopping List</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {Object.entries(shoppingList)
                      .filter(([category, items]) => items.length > 0)
                      .map(([category, items]) => (
                        <div key={category} className="bg-white p-6 rounded-xl shadow-lg">
                          <h4 className="text-xl font-semibold text-red-600 mb-4 capitalize border-b-2 border-gray-200 pb-2">
                            {category === 'nutsSeeds' ? 'Nuts & Seeds' :
                             category === 'grains' ? 'Grains & Carbs' :
                             category.charAt(0).toUpperCase() + category.slice(1)}
                          </h4>
                          <ul className="space-y-2">
                            {items.map((item, index) => (
                              <li key={index} className="flex items-center text-gray-700">
                                <span className="text-green-500 mr-3">✅</span>
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              <div className="flex justify-center gap-6 mt-12">
                <button
                  onClick={() => window.print()}
                  className="bg-gradient-to-r from-green-500 to-green-600 text-white py-4 px-8 rounded-2xl text-xl font-semibold hover:from-green-600 hover:to-green-700 transform hover:scale-105 transition-all duration-200"
                >
                  Print Plan
                </button>
                <button
                  onClick={handleNewAssessment}
                  className="bg-gradient-to-r from-gray-500 to-gray-600 text-white py-4 px-8 rounded-2xl text-xl font-semibold hover:from-gray-600 hover:to-gray-700 transform hover:scale-105 transition-all duration-200"
                >
                  New Assessment
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {currentPage === 'profile' ? renderProfilePage() : renderMealPlanPage()}

      {/* Popup Modal */}
      {showPopup && assessment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-8 rounded-3xl text-center max-w-md w-full mx-4">
            <button
              onClick={() => setShowPopup(false)}
              className="absolute top-4 right-4 text-2xl text-gray-500 hover:text-gray-700"
            >
              &times;
            </button>
            <div className="text-6xl mb-4">{assessment.healthStatus === 'healthy' ? '😊' : '⚠️'}</div>
            <h2 className="text-3xl font-bold mb-4">
              {assessment.healthStatus === 'healthy' ? 'Great News!' : 'Health Assessment Complete'}
            </h2>
            <p className="text-lg mb-4">
              {assessment.healthStatus === 'healthy'
                ? 'Your health is in check! Keep up the good work with your current lifestyle.'
                : `We've identified some areas for improvement in your nutrition. Diagnosis: ${assessment.diagnosis}`
              }
            </p>
            {assessment.healthStatus !== 'healthy' && (
              <button
                onClick={handleViewMealPlan}
                className="bg-gradient-to-r from-red-500 to-red-600 text-white py-3 px-6 rounded-2xl text-lg font-semibold hover:from-red-600 hover:to-red-700 transform hover:scale-105 transition-all duration-200"
              >
                View Meal Plan
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default App;
