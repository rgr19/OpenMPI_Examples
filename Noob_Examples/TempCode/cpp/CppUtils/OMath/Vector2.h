#pragma once

#include "../Math/math.h"

class OVector2 {
		friend class OMatrix4;

	public:
		//constructors
		OVector2( );

		OVector2( float, float );

		OVector2( const OVector2& );

		~OVector2( );

		//access functions
		OVector2& operator =( const OVector2& );

		float& operator []( int );

		inline float& X( ) {
			return x_;
		}

		inline float& Y( ) {
			return y_;
		}

		void Print( );

		//math operators
		OVector2 operator +( const OVector2& ) const;

		OVector2 operator -( const OVector2& ) const;

		OVector2 operator *( const float& ) const;

		float operator *( const OVector2& ) const;

		inline const float* GetSafeV() {return vec_;}


		float Dot( const OVector2& ) const;

		inline float FastNorm( ) const {
			return SQRT( x_ * x_ + y_ * y_ );
		}

		inline float Norm( ) const {
			return sqrt( x_ * x_ + y_ * y_ );
		}

		// 1 / FastNorm
		inline float FastInvertNorm( ) {
			return RSQRT( x_ * x_ + y_ * y_ );
		}

		bool normalize( );

		bool shortest_rotation( OVector2 axis_1, OVector2 axis_2 );



	protected:
		//data
		union {
			struct {
				float x_;
				float y_;
			};
			float vec_[2];
		};

};

OVector2 operator *( const float&, const OVector2& );
